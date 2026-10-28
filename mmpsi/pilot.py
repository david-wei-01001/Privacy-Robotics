"""Night-1 pilot orchestrator (PLAN §9.2). Stages are resumable; a failing stage or encoder is logged and skipped.

  python -m mmpsi.pilot --config configs/pilot_n1.yaml --night 2026-10-27_n1 [--smoke]
  python -m mmpsi.pilot --night 2026-10-27_n1 --report-only      # re-render the reports from facts.json
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import subprocess
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import yaml

from . import features  # noqa: F401  (registers encoders)
from . import matching  # noqa: F401  (registers matchers)
from .core import REPO, Heartbeat, StageLog, build, data_root, get_logger, git_state, plan_version, short_hash
from .data import _h, build_sins, build_tv_mix, build_xd
from .evaluate import dataset_comparisons, evaluate_one, track_masks
from .matching import Prepared, apple_psi_equivalence, l2n
from .report import comparison_md, derive, feasibility_md, lint_report, roc_figure, tables_and_figures

SMOKE = {"xd": {"max_violent": 6, "max_normal": 4}, "sins": {"max_files": 120}, "tv_mix": {"n": 12},
         "budget": {"extract_s": 900, "probe_windows": 8}}


def deep_update(d, u):
    for k, v in u.items():
        d[k] = deep_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def ensure_sins_archives(root, cfg, log):
    raw = root / "raw/sins"
    for i in cfg["sins"]["archives"]:
        f = raw / f"DCASE2018-task5-dev.audio.{i}.zip"
        for attempt in range(6):
            if f.exists() and zipfile.is_zipfile(f):
                break
            log.info(f"SINS: downloading/resuming {f.name} (attempt {attempt + 1})")
            subprocess.run(["curl", "-sL", "-C", "-", "--retry", "5", "-o", str(f),
                            f"https://zenodo.org/records/1247090/files/{f.name}?download=1"], check=False)


def counts(W):
    out = {}
    for (ds, split, label), g in W.groupby(["dataset", "split", "label"]):
        out[f"{ds}/{split}/{label}"] = {"windows": int(len(g)), "groups": int(g.group_id.nunique()),
                                        "events": int(g.event_id.nunique())}
    return out


def scale_windows(W, encs, sec_per_window, budget_s, log):
    est = sum(sec_per_window[e] * applicable(W, encs[e]).sum() for e in encs)
    if est <= budget_s:
        return W, 1.0, est
    frac = max(0.15, 0.9 * budget_s / est)
    shrink = W.dataset.isin(["sins"]) | ((W.dataset == "xd") & (W.label == "benign"))
    keep = ~shrink | (W.group_id.map(lambda g: _h(str(g), "scale")) < frac)
    log.info(f"scaling benign windows to {frac:.2f} to fit the extraction budget (estimate {est / 3600:.1f} h)")
    return W[keep].reset_index(drop=True), frac, est


def applicable(W, enc) -> np.ndarray:
    if enc.modality == "audio":
        return W.has_audio.values.astype(bool)
    return (W.dataset == "xd").values & (W.video_path.values != "")


def probe_sample(W: pd.DataFrame, enc, n: int) -> pd.DataFrame:
    """Windows for the timing probe: n random windows; for a video encoder, every window of two videos, because it
    decodes whole videos, so its cost per window only shows over a whole video."""
    sub = W[applicable(W, enc)]
    if enc.modality == "video":
        vids = pd.Series(sub.video_path.unique()).sample(min(2, sub.video_path.nunique()), random_state=0)
        return sub[sub.video_path.isin(vids)]
    return sub.sample(min(n, len(sub)), random_state=0)


def save_rocs(rocs: dict, path):
    np.savez(path, **{f"{k}__y": y for k, (y, _) in rocs.items()}, **{f"{k}__s": s for k, (_, s) in rocs.items()})


def load_rocs(path) -> dict:
    if not path.exists():
        return {}
    z = np.load(path)
    return {k[:-3]: (z[k], z[k[:-3] + "__s"]) for k in z.files if k.endswith("__y")}


def render(facts: dict, rep, rocs: dict):
    """Derived comparisons, tables, figures, comparison.md (then its check), feasibility.md, and facts.json."""
    facts["derived"] = derive(facts)
    tables_and_figures(facts, rep)
    if rocs:
        roc_figure(rocs, rep / "figures" / "roc_home", order=list(facts.get("encoders", {})))
    comparison = comparison_md(facts)
    facts["report_lint"] = lint_report(comparison, facts)
    (rep / "comparison.md").write_text(comparison)
    (rep / "feasibility.md").write_text(feasibility_md(facts))
    (rep / "facts.json").write_text(json.dumps(facts, indent=2, default=str))


def append_results(facts: dict, cfg: dict, log):
    """One results.tsv row per run (PLAN §8.2), the dataset-setting runs included. A rerun of the same night does
    not append its rows twice."""
    tsv = REPO / "autoresearch/results.tsv"
    seen = {line.split("\t", 1)[0] for line in tsv.read_text().splitlines()[1:]}
    ds = facts.get("datasets") or {}
    rows = [dict(r, note=r["track"]) for r in facts["runs"]]
    rows += [dict(r, track="home", encoder=ds["encoder"], note=f"datasets/{r['setting']}")
             for rs in ds.get("settings", {}).values() for r in rs]
    new = [r for r in rows if r["run_id"] not in seen]
    if len(new) < len(rows):
        log.info(f"results.tsv: {len(rows) - len(new)} of these runs are already logged; appending {len(new)} rows")
    rows = new
    with tsv.open("a") as f:
        for r in rows:
            f.write("\t".join(map(str, [r["run_id"], time.strftime("%Y-%m-%dT%H:%M"), facts["meta"]["plan_version"],
                                        facts["meta"]["git"], "pilot", facts["meta"]["config_hash"], "xd+sins+tv_mix",
                                        r["encoder"], "", "", "l2norm+center", r["matcher"] if r["matcher"] == "lsh" else "",
                                        f"{r['matcher']}{r['params'] or ''}", cfg["window_s"], 1, "", r["fmr_target"],
                                        round(r["fmr_check"], 5), round(r["recall_event"], 4), round(r["recall_window"], 4),
                                        round(r["auc"], 4), "", "", "", "", "pilot", r["note"]])) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_n1.yaml")
    ap.add_argument("--night", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-render the reports from an existing facts.json")
    a = ap.parse_args()
    root = data_root()
    logs, runs_dir, rep = root / "logs" / a.night, REPO / "runs" / a.night, REPO / "reports" / a.night
    for d in (logs, runs_dir, rep):
        d.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        facts = json.loads((rep / "facts.json").read_text())
        # the report code may be newer than the run's code copy, so keep the version that rendered these reports
        (runs_dir / "code").mkdir(exist_ok=True)
        shutil.copy2(REPO / "mmpsi/report.py", runs_dir / "code" / "report_rerender.py")
        facts["meta"].update(rerendered=time.strftime("%Y-%m-%d %H:%M"), rerender_code=f"runs/{a.night}/code/report_rerender.py")
        render(facts, rep, load_rocs(runs_dir / "roc.npz"))
        print(f"reports re-rendered in {rep}")
        return
    cfg = yaml.safe_load((REPO / a.config).read_text())
    if a.smoke:
        cfg = deep_update(copy.deepcopy(cfg), SMOKE)
    log = get_logger(logs / "pilot.log")
    hb, st = Heartbeat(logs / "heartbeat.json"), StageLog(runs_dir / "stages.json")
    # nothing is committed yet, so the run keeps a copy of the code and config it ran with
    src = sorted((REPO / "mmpsi").glob("*.py")) + [REPO / a.config]
    (runs_dir / "code").mkdir(exist_ok=True)
    for p in src:
        shutil.copy2(p, runs_dir / "code" / p.name)
    facts = {"meta": {"night": a.night, "plan_version": plan_version(), "git": git_state(), "smoke": a.smoke,
                      "code_hash": short_hash([p.read_text() for p in src]), "code_copy": f"runs/{a.night}/code/",
                      "config": cfg, "config_hash": short_hash(cfg), "started": time.strftime("%Y-%m-%d %H:%M")},
             "runs": [], "stages": st.stages}
    ids = itertools.count(1)

    def new_id() -> str:  # run ids follow PLAN §13.2: <night_id>-<NNNN>
        return f"{a.night}-{next(ids):04d}"

    def save():  # facts.json after every stage, so a crash later in the night keeps what is done
        facts["stages"] = st.stages
        (rep / "facts.json").write_text(json.dumps(facts, indent=2, default=str))

    t0 = time.time()

    # ---------------------------------------------------------------- 1. data
    hb.set("data")
    st.start("data")
    wpath = root / "derived" / f"windows_{short_hash({k: cfg[k] for k in ('xd', 'sins', 'tv_mix', 'window_s', 'hop_s')})}.csv"
    try:
        if wpath.exists():
            W = pd.read_csv(wpath, keep_default_na=False)
            W["has_audio"] = W.has_audio.astype(str).str.lower().isin(["true", "1"])
        else:
            ensure_sins_archives(root, cfg, log)
            xd = build_xd(cfg, root, log)
            sins = build_sins(cfg, root, log)
            W = pd.concat([xd, sins], ignore_index=True)
            W = pd.concat([W, build_tv_mix(cfg, root, W, log)], ignore_index=True)
            W["window_id"] = W.dataset + "/" + W.clip_id + "/" + W.t0.map(lambda t: f"{t:.2f}")
            W.to_csv(wpath, index=False)
        facts["counts"] = counts(W)
        st.done("data", windows=len(W), file=str(wpath))
        save()
    except Exception as e:  # noqa: BLE001
        st.done("data", "failed", error=f"{type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        save()
        return

    # ---------------------------------------------------------------- 2. probe and scale
    hb.set("probe")
    st.start("probe")
    encs, spw = {}, {}
    for name in cfg["encoders"]:
        try:
            e = build("encoder", name)
            sample = probe_sample(W, e, cfg["budget"]["probe_windows"])
            e.load()
            e.encode(sample.head(2))  # warm-up: model load, first MPS kernels, and file reads are not timed
            t = time.time()
            e.encode(sample)
            spw[name] = (time.time() - t) / max(1, len(sample))
            encs[name] = e
            log.info(f"probe {name}: {1000 * spw[name]:.0f} ms/window ({len(sample)} windows)")
        except Exception as ex:  # noqa: BLE001
            log.warning(f"encoder {name} unavailable: {type(ex).__name__}: {ex}")
            log.error(traceback.format_exc())
            st.stages.setdefault(f"encoder:{name}", {}).update(status="unavailable", error=f"{type(ex).__name__}: {ex}")
    W, frac, est = scale_windows(W, encs, spw, cfg["budget"]["extract_s"], log)
    facts["counts"] = counts(W)
    facts["sins_files"] = {"available": len(list((root / "derived/sins/audio16k").glob("*.wav"))),
                           "used": int(W[W.dataset == "sins"].clip_id.nunique())}
    facts["scale"] = {"benign_fraction_kept": frac, "estimated_extract_s": round(float(est)),
                      "ms_per_window": {k: round(1000 * v, 1) for k, v in spw.items()}}
    facts["encoders"] = {k: {"modality": v.modality, "neural": v.neural, "dim": v.dim} for k, v in encs.items()}
    st.done("probe", encoders=list(encs))
    save()

    # ---------------------------------------------------------------- 3. extract
    whash = short_hash(W.window_id.tolist())
    E, order, t_extract = {}, sorted(encs, key=lambda n: spw[n]), time.time()
    for i, name in enumerate(order):
        enc = encs[name]
        hb.set(f"extract:{name}")
        st.start(f"encoder:{name}")
        cpath = root / "cache/emb" / f"{whash}_{enc.fingerprint()}.npy"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            cov, note = 1.0, "cached"
            if cpath.exists():
                E[name] = np.load(cpath)
            else:
                left = cfg["budget"]["extract_s"] - (time.time() - t_extract)
                app = applicable(W, enc)
                arr = np.full((len(W), enc.dim), np.nan, np.float32)
                arr[app] = enc.encode(W[app].reset_index(drop=True), budget_s=max(300, left / (len(order) - i)) * 1.5, log=log)
                cov, note = getattr(enc, "coverage", 1.0), ""
                # a partial result is kept for tonight but not cached, so a rerun with more time completes it
                np.save(cpath if cov >= 1 else cpath.with_name(cpath.stem + "_partial.npy"), arr)
                E[name] = arr
            n_ok = int(np.isfinite(E[name][:, 0]).sum())
            facts["encoders"][name].update(windows=n_ok, coverage=round(cov, 3))
            st.done(f"encoder:{name}", "ok" if cov >= 1 else "partial", windows=n_ok, coverage=round(cov, 3), note=note)
        except Exception as ex:  # noqa: BLE001
            st.done(f"encoder:{name}", "failed", error=f"{type(ex).__name__}: {ex}")
            log.error(traceback.format_exc())
        save()

    # ---------------------------------------------------------------- 4. evaluate
    hb.set("evaluate")
    st.start("evaluate")
    rocs, P_cache = {}, {}
    try:
        audio = [n for n in E if encs[n].modality == "audio"]
        for track in ("home", "xd"):
            names = list(audio) + ([n for n in E if encs[n].modality == "video"] if track == "xd" else [])
            if track == "xd" and audio and "clip_frames" in E:
                best_a = max(audio, key=lambda n: max([r["recall_event"] for r in facts["runs"] if r["encoder"] == n
                                                       and r["matcher"] == "cosine_top1" and r["fmr_target"] == 0.01] or [0]))
                fused = np.concatenate([l2n(np.nan_to_num(E[best_a])), l2n(np.nan_to_num(E["clip_frames"]))], 1)
                ok = np.isfinite(E[best_a][:, 0]) & np.isfinite(E["clip_frames"][:, 0])
                fused[~ok] = np.nan
                E[f"fusion:{best_a}+clip_frames"] = fused
                names.append(f"fusion:{best_a}+clip_frames")
            for name in names:
                mod = "audio" if name in audio else "video"
                masks = track_masks(W, track, mod)
                fin = np.isfinite(E[name][:, 0])
                masks = {k: v & fin for k, v in masks.items()}
                if masks["ref"].sum() < 2 or masks["calib"].sum() < 10:
                    log.warning(f"evaluate {track}/{name}: too few windows (ref {masks['ref'].sum()}, calib {masks['calib'].sum()})")
                    continue
                P = Prepared(np.nan_to_num(E[name]), masks["ref"], masks["benign_train"])
                if track == "home":
                    P_cache[(track, name)] = (P, masks)
                for spec in cfg["matchers"]:
                    if spec["name"] == "logreg" and not masks["benign_train"].any():
                        log.warning(f"evaluate {track}/{name}: logreg skipped (no benign_train windows to train on)")
                        continue
                    try:
                        rows, s_all = evaluate_one(W, P, masks, spec, new_id)
                        for r in rows:
                            r.update(track=track, encoder=name)
                        facts["runs"] += rows
                        if track == "home" and spec["name"] == "cosine_top1":
                            v, c = masks["violent"], masks["check"]
                            rocs[name] = (np.r_[np.ones(v.sum()), np.zeros(c.sum())], np.r_[s_all[v], s_all[c]])
                    except Exception as ex:  # noqa: BLE001
                        log.warning(f"evaluate {track}/{name}/{spec}: {type(ex).__name__}: {ex}")
                        log.error(traceback.format_exc())
        if rocs:
            save_rocs(rocs, runs_dir / "roc.npz")
        st.done("evaluate", runs=len(facts["runs"]))
    except Exception as ex:  # noqa: BLE001
        st.done("evaluate", "failed", error=f"{type(ex).__name__}: {ex}")
        log.error(traceback.format_exc())
    save()

    runs = pd.DataFrame(facts["runs"])
    best = None
    if len(runs):
        h = runs[(runs.track == "home") & (runs.matcher == "cosine_top1") & (runs.fmr_target == 0.01)]
        best = h.sort_values("recall_event", ascending=False).encoder.iloc[0] if len(h) else None

    # ---------------------------------------------------------------- 5. dataset comparisons
    hb.set("datasets")
    st.start("datasets")
    try:
        if best and ("home", best) in P_cache:
            P, mh = P_cache[("home", best)]
            fin = np.isfinite(E[best][:, 0])
            mx = {k: v & fin for k, v in track_masks(W, "xd", "audio").items()}
            facts["datasets"] = dict(encoder=best, matcher="cosine_top1",
                                     **dataset_comparisons(W, P, mh, mx, {"name": "cosine_top1"}, new_id))
        st.done("datasets", encoder=best)
    except Exception as ex:  # noqa: BLE001
        st.done("datasets", "failed", error=f"{type(ex).__name__}: {ex}")
        log.error(traceback.format_exc())
    save()

    # ---------------------------------------------------------------- 6. Apple PSI equivalence
    hb.set("apple_psi")
    st.start("apple_psi")
    try:
        if best and ("home", best) in P_cache:
            P, mh = P_cache[("home", best)]
            lsh = runs[(runs.track == "home") & (runs.encoder == best) & (runs.matcher == "lsh") & (runs.fmr_target == 0.01)]
            if len(lsh):
                r = lsh.sort_values(["recall_event", "fmr_check"], ascending=[False, True]).iloc[0]
                # half violent, half benign windows, so that both server decisions occur
                rng, n = np.random.default_rng(0), cfg["budget"].get("psi_windows", 150)
                viol, ben = np.where(mh["violent"])[0], np.where(mh["check"])[0]
                nv = min(len(viol), n // 2)
                q = np.r_[rng.choice(viol, nv, replace=False), rng.choice(ben, min(len(ben), n - nv), replace=False)]
                m = int(min(r.tau, r.params["L"]))
                facts["equivalence"] = dict(run_id=new_id(), encoder=best, lsh_run_id=r.run_id,
                                            **apple_psi_equivalence(P, mh["ref"], q, int(r.params["k"]), int(r.params["L"]),
                                                                    max(1, m), log=log))
        st.done("apple_psi", **{k: v for k, v in (facts.get("equivalence") or {}).items() if k != "run_id"})
    except Exception as ex:  # noqa: BLE001
        st.done("apple_psi", "failed", error=f"{type(ex).__name__}: {ex}")
        log.error(traceback.format_exc())
    save()

    # ---------------------------------------------------------------- 7. report
    hb.set("report")
    st.start("report")
    try:
        facts["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M")
        facts["meta"]["minutes"] = round((time.time() - t0) / 60, 1)
        st.done("report")
        save()
        render(facts, rep, rocs)
        if not a.smoke and len(runs):
            append_results(facts, cfg, log)
        (runs_dir / "summary.json").write_text(json.dumps({"meta": facts["meta"], "stages": st.stages,
                                                           "n_runs": len(facts["runs"])}, indent=2, default=str))
    except Exception as ex:  # noqa: BLE001
        st.done("report", "failed", error=f"{type(ex).__name__}: {ex}")
        log.error(traceback.format_exc())
        save()
    hb.set("done")
    log.info(f"pilot finished in {(time.time() - t0) / 60:.1f} min; reports in {rep}")


if __name__ == "__main__":
    main()
