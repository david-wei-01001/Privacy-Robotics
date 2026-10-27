"""Report generation (PLAN §12): derived comparisons, tables, figures, feasibility.md, and comparison.md.

comparison.md follows Chenxinran's bullet rules (PLAN §12.2): one comparison per bullet, claim first, "we", hedged
interpretations, abbreviations defined on first use, no em-dashes. Every number in it comes from facts.json (a run, or
a derived comparison that derive() stores there with its run ids), and lint_report() checks this (PLAN §12.1 step 3).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOUND_EVENT, SPEECH = ("vggish", "ast", "clap"), ("hubert",)
# chart colors (dataviz reference palette): categorical slots in fixed order, a one-hue blue ramp, recessive chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d=1):
    return "n/a" if _nan(x) else f"{100 * x:.{d}f}%"


def ci(c):
    return "n/a" if c is None or any(_nan(v) for v in c) else f"{100 * c[0]:.1f} to {100 * c[1]:.1f}"


def f3(x):
    return "n/a" if _nan(x) else f"{x:.3f}"


def _enum(items: list[str]) -> str:
    """Numbered inline list, "1) a, 2) b, and 3) c" (PLAN §12.2)."""
    items = [f"{i + 1}) {s}" for i, s in enumerate(items)]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


def _runs_df(facts):
    df = pd.DataFrame(facts["runs"])
    if len(df):
        df["params_s"] = df.params.apply(lambda p: ", ".join(f"{k} = {v}" for k, v in sorted(p.items(), key=lambda kv: kv[0].lower())))
        df["matcher_full"] = df.matcher + df.params_s.apply(lambda s: f" ({s})" if s else "")
    return df


def _row(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0] if len(df) else None


# ============================================================================ derived comparisons
def derive(facts: dict) -> dict:
    """Gaps and ratios that the report states, each with the run ids it is computed from (stored in facts.json)."""
    df = _runs_df(facts)
    out = {"best_encoder": {}, "discretization_gap": {}, "classifier_gap": {}, "baseline_gap": {}, "tv_ratio": {}}
    if not len(df):
        return out
    h = df[df.track == "home"]
    for fmr in sorted(h.fmr_target.unique(), reverse=True):
        key = f"{fmr:g}"
        cos = h[(h.matcher == "cosine_top1") & (h.fmr_target == fmr)].sort_values("recall_event", ascending=False)
        if not len(cos):
            continue
        b = cos.iloc[0]
        out["best_encoder"][key] = dict(encoder=b.encoder, recall_event=b.recall_event, run_id=b.run_id)
        mf = _row(cos, encoder="mfcc_stats")
        for r in cos.itertuples():
            lsh = h[(h.encoder == r.encoder) & (h.matcher == "lsh") & (h.fmr_target == fmr)]
            if len(lsh):
                l1 = lsh.sort_values(["recall_event", "fmr_check"], ascending=[False, True]).iloc[0]
                out["discretization_gap"].setdefault(r.encoder, {})[key] = dict(
                    cosine_run=r.run_id, cosine_recall=r.recall_event, lsh_run=l1.run_id, lsh_recall=l1.recall_event,
                    lsh_params=l1.params, m=l1.tau, gap_points=round(100 * (r.recall_event - l1.recall_event), 1))
            g = _row(h, encoder=r.encoder, matcher="logreg", fmr_target=fmr)
            if g is not None:
                out["classifier_gap"].setdefault(r.encoder, {})[key] = dict(
                    cosine_run=r.run_id, cosine_recall=r.recall_event, logreg_run=g.run_id, logreg_recall=g.recall_event,
                    gap_points=round(100 * (g.recall_event - r.recall_event), 1))
            if mf is not None and r.encoder != "mfcc_stats":
                a, c = r.recall_event_ci, mf.recall_event_ci
                out["baseline_gap"].setdefault(r.encoder, {})[key] = dict(
                    run=r.run_id, recall=r.recall_event, mfcc_run=mf.run_id, mfcc_recall=mf.recall_event,
                    gap_points=round(100 * (r.recall_event - mf.recall_event), 1),
                    ci_overlap=bool(a[0] <= c[1] and c[0] <= a[1]))
            if key == "0.01" and not _nan(r.tv_sins_fmr) and r.fmr_check > 0:
                out["tv_ratio"][r.encoder] = dict(run=r.run_id, tv_sins_fmr=r.tv_sins_fmr, fmr_check=r.fmr_check,
                                                  ratio=round(r.tv_sins_fmr / r.fmr_check, 1))
    return out


# ============================================================================ tables and figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def _chrome(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.savefig(path.parent / f"{path.name}.png", dpi=200)
    fig.savefig(path.parent / f"{path.name}.pdf")


def fmr_tag(fmr: float) -> str:
    """'fmr1pct', 'fmr0p1pct': no dot, so file names keep their suffix."""
    return f"fmr{100 * fmr:g}pct".replace(".", "p")


def tables_and_figures(facts: dict, out: Path):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = _runs_df(facts)
    if not len(df):
        return
    df.drop(columns=["params"]).to_csv(out / "tables" / "runs.csv", index=False)
    order = list(facts.get("encoders", {}))
    for track in df.track.unique():
        for fmr in sorted(df.fmr_target.unique(), reverse=True):
            d = df[(df.track == track) & (df.fmr_target == fmr)]
            piv = d.pivot_table(index="encoder", columns="matcher_full", values="recall_event", aggfunc="first", sort=False)
            piv = piv.reindex([e for e in order if e in piv.index] + [e for e in piv.index if e not in order])
            tag = fmr_tag(fmr)
            (out / "tables" / f"recall_event_{tag}_{track}.md").write_text(piv.map(pct).to_markdown())
            try:
                heatmap(piv, f"Event recall at FMR {100 * fmr:g}% ({track} track)", out / "figures" / f"heatmap_{track}_{tag}")
            except Exception as e:  # noqa: BLE001
                (out / "figures" / f"heatmap_{track}_{tag}.error.txt").write_text(str(e))
    try:
        recall_figure(df, order, out / "figures" / "recall_by_encoder_home")
    except Exception as e:  # noqa: BLE001
        (out / "figures" / "recall_by_encoder_home.error.txt").write_text(str(e))


def heatmap(piv: pd.DataFrame, title: str, path: Path):
    """Encoder x matcher grid of event recall: one-hue sequential ramp, values printed in ink (a table view)."""
    from matplotlib.colors import LinearSegmentedColormap
    plt = _plt()
    vals = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(0.62 * len(piv.columns) + 2.2, 0.36 * len(piv) + 1.9))
    im = ax.imshow(vals, vmin=0, vmax=1, cmap=LinearSegmentedColormap.from_list("blue", BLUES), aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=50, ha="right")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not math.isnan(vals[i, j]):
                ax.text(j, i, f"{100 * vals[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if vals[i, j] >= 0.55 else INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def recall_figure(df: pd.DataFrame, order: list[str], path: Path):
    """Event recall with 95% CI per encoder (home track, cosine top-1), one dot series per FMR target."""
    plt = _plt()
    from matplotlib.ticker import PercentFormatter
    h = df[(df.track == "home") & (df.matcher == "cosine_top1")]
    encs = [e for e in order if e in set(h.encoder)]
    if not encs:
        return
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(encs) + 1.3))
    for j, fmr in enumerate(sorted(h.fmr_target.unique(), reverse=True)):
        d = h[h.fmr_target == fmr].set_index("encoder")
        x = np.array([d.recall_event.get(e, np.nan) for e in encs], float)
        lo = np.array([d.recall_event_ci[e][0] if e in d.index else np.nan for e in encs], float)
        hi = np.array([d.recall_event_ci[e][1] if e in d.index else np.nan for e in encs], float)
        y = np.arange(len(encs)) + (j - 0.5) * 0.3
        ax.errorbar(x, y, xerr=[np.maximum(x - lo, 0), np.maximum(hi - x, 0)], fmt="o", ms=5, color=SERIES[j],
                    elinewidth=1.4, capsize=0, label=f"FMR target {100 * fmr:g}%")
    ax.set_yticks(range(len(encs)), encs)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("event recall on validation events (95% CI, bootstrap over events)")
    ax.set_title("Event recall by encoder (home track, cosine top-1)", fontsize=9, color=INK, loc="left")
    _chrome(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def roc_figure(curves: dict, path: Path, order: list[str] | None = None):
    """Window-level ROC per encoder; colors follow the encoder's position in the config, not its rank."""
    try:
        plt = _plt()
        from sklearn.metrics import roc_curve
        order = order or list(curves)
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        for name in sorted(curves, key=lambda n: order.index(n) if n in order else len(order)):
            y, s = curves[name]
            fpr, tpr, _ = roc_curve(y, s)
            i = order.index(name) if name in order else len(order)
            ax.plot(fpr, tpr, lw=1.4, color=SERIES[i % len(SERIES)], label=name)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.01)
        for f in (0.01, 0.001):
            ax.axvline(f, ls=":", c=MUTED, lw=0.8)
        ax.set_xlabel("benign false-match rate (window, SINS val_check)")
        ax.set_ylabel("violent recall (window)")
        ax.set_title("ROC per encoder (home track, cosine top-1)", fontsize=9, color=INK, loc="left")
        _chrome(ax)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="lower right")
        fig.tight_layout()
        _save(fig, path)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        (path.parent / f"{path.name}.error.txt").write_text(str(e))


# ============================================================================ feasibility.md
def feasibility_md(facts: dict) -> str:
    st, cnt, df, meta = facts.get("stages", {}), facts.get("counts", {}), _runs_df(facts), facts["meta"]
    dv = facts.get("derived") or derive(facts)
    L = [f"# Night {meta['night']}: feasibility", "",
         f"- Plan version {meta['plan_version']}; code {meta['git']} (copy in {meta.get('code_copy', 'n/a')}); started {meta.get('started')}, finished "
         f"{meta.get('finished')} ({meta.get('minutes')} minutes)."]
    if meta.get("smoke"):
        L.append("- Smoke test on tiny subsets: the rows below only show that each check can be computed.")
    L += ["", "## 1. Did each stage run?", "", "| Stage | Status | Minutes | Notes |", "|---|---|---|---|"]
    for name, s in st.items():
        note = str(s.get("error") or s.get("note") or "")
        if name.startswith("encoder:") and "windows" in s:
            cov = s.get("coverage", 1)
            note = f"{s['windows']} windows" + (f", {100 * cov:.0f}% of the planned windows" if cov < 1 else "") + (f"; {note}" if note else "")
        L.append(f"| {name} | {s.get('status')} | {round(s.get('seconds', 0) / 60, 1)} | {note[:200]} |")
    sc = facts.get("scale") or {}
    if sc:
        budget = meta["config"]["budget"]["extract_s"]
        L += ["", f"- Timing probe, ms per window: " + ", ".join(f"{k} {v}" for k, v in sc["ms_per_window"].items())
              + f". Estimated extraction: {sc['estimated_extract_s'] / 3600:.1f} h for a budget of {budget / 3600:.1f} h; "
              f"share of benign windows kept: {100 * sc['benign_fraction_kept']:.0f}%."]

    L += ["", "## 2. Data at this scale", "", "| Set (dataset/split/label) | Windows | Groups | Events |", "|---|---|---|---|"]
    for k, v in cnt.items():
        L.append(f"| {k} | {v.get('windows')} | {v.get('groups')} | {v.get('events')} |")

    L += ["", "## 3. Are the metrics computable, and how wide are the CIs?", "",
          "| Metric | Computed | At this scale |", "|---|---|---|"]
    crit, h = {}, (df[df.track == "home"] if len(df) else df)
    c1 = h[(h.matcher == "cosine_top1") & (h.fmr_target == 0.01)] if len(h) else h
    c01 = h[(h.matcher == "cosine_top1") & (h.fmr_target == 0.001)] if len(h) else h

    def halfwidths(rows, col):
        return [(c[1] - c[0]) / 2 for c in rows[col] if c is not None and not any(_nan(x) for x in c)]

    if len(c1):
        w1, w01, wf = halfwidths(c1, "recall_event_ci"), halfwidths(c01, "recall_event_ci"), halfwidths(c1, "fmr_check_ci")
        n_chk, n_ev = int(c1.n_check.max()), int(c1.n_events.max())
        L.append(f"| event recall at FMR 1% (cosine, home) | {len(c1)} encoders | {n_ev} validation events; 95% CI half-width "
                 f"median {100 * np.median(w1):.1f}, max {100 * max(w1):.1f} points |" if w1 else "| event recall at FMR 1% | no CI | |")
        L.append(f"| event recall at FMR 0.1% | {len(c01)} encoders | 95% CI half-width median {100 * np.median(w01):.1f} points |"
                 if w01 else "| event recall at FMR 0.1% | no CI | |")
        L.append(f"| measured FMR on val_check (1% target) | yes | {n_chk} benign windows; values "
                 f"{pct(c1.fmr_check.min(), 2)} to {pct(c1.fmr_check.max(), 2)}; 95% CI half-width (by session) up to "
                 f"{100 * max(wf):.2f} points |" if wf else "| measured FMR | no CI | |")
        L.append(f"| measured FMR at the 0.1% target | yes | {n_chk * 0.001:.1f} expected false matches in val_check "
                 "(at least 3 are needed for a usable estimate) |")
        L.append(f"| AUC | yes | {f3(c1.auc.min())} to {f3(c1.auc.max())} |")
        L.append(f"| FMR on TV mix / SINS watching TV (1% target) | {'yes' if c1.tv_fmr.notna().any() else 'no'} | TV mix "
                 f"{pct(c1.tv_fmr.min(), 2)} to {pct(c1.tv_fmr.max(), 2)}; watching TV {pct(c1.tv_sins_fmr.min(), 2)} to "
                 f"{pct(c1.tv_sins_fmr.max(), 2)} |")
        crit["metrics: recall at FMR 1% has a 95% CI no wider than +/-10 points"] = (
            bool(w1) and max(w1) <= 0.10, f"max half-width {100 * max(w1):.1f} points" if w1 else "n/a")
        crit["metrics: val_check holds at least 3,000 benign windows"] = (n_chk >= 3000, f"{n_chk} windows")
    for kind, label in (("classifier_gap", "classifier gap (logreg minus cosine)"),
                        ("discretization_gap", "discretization gap (cosine minus best LSH)")):
        g = [v["0.01"]["gap_points"] for v in dv.get(kind, {}).values() if "0.01" in v]
        L.append(f"| {label}, FMR 1% | {len(g)} encoders | {min(g):.1f} to {max(g):.1f} points |" if g else f"| {label} | no | |")
    eq = facts.get("equivalence") or {}
    if eq:
        L.append(f"| Apple PSI vs plaintext LSH | yes | equivalence {pct(eq['equivalence_rate'])} on {eq['n_windows']} windows "
                 f"({eq['n_plain_positive']} plaintext matches); cuckoo drops {eq['cuckoo_dropped']} of {eq['n_items_X']} items; "
                 f"{eq['n_disagree']} disagreements, {eq['n_disagree_explained_by_drops']} explained by drops; voucher "
                 f"{eq['voucher_bytes']} bytes; {eq['client_ms_per_voucher']} ms per voucher (robot), "
                 f"{eq['server_ms_per_voucher']} ms (server); setup {eq['setup_s']} s |")
    L.append("| runtime per stage | yes | section 1 |")

    # ---- go / no-go (PLAN §9.2)
    if len(df):
        audio_ok = sorted(set(h.encoder)) if len(h) else []
        crit["pipeline: at least 3 audio encoders completed"] = (len(audio_ok) >= 3, ", ".join(audio_ok) or "none")
        fam = {"cosine_top1", "cosine_top5", "l2_raw", "hamming_sign", "lsh", "logreg"}
        crit["pipeline: every similarity family ran"] = (set(df.matcher) >= fam, ", ".join(sorted(fam - set(df.matcher))) or "all ran")
    if eq:
        crit["pipeline: Apple PSI decisions match plaintext LSH, except cuckoo drops"] = (
            eq["n_disagree"] == eq["n_disagree_explained_by_drops"], f"{eq['n_disagree']} disagreements, "
            f"{eq['n_disagree_explained_by_drops']} explained by drops")
    bg = {e: v["0.01"] for e, v in dv.get("baseline_gap", {}).items() if "0.01" in v}
    if bg:
        e, v = max(bg.items(), key=lambda kv: -1e9 if _nan(kv[1]["gap_points"]) else kv[1]["gap_points"])
        crit["promising: an encoder clearly exceeds MFCC at FMR 1% (read as: higher, and 95% CIs do not overlap)"] = (
            v["gap_points"] > 0 and not v["ci_overlap"], f"best {e}: {v['gap_points']:+.1f} points, CIs "
            f"{'overlap' if v['ci_overlap'] else 'do not overlap'}")
    dg = {e: v["0.01"] for e, v in dv.get("discretization_gap", {}).items() if "0.01" in v}
    if dg:
        e, v = min(dg.items(), key=lambda kv: 1e9 if _nan(kv[1]["gap_points"]) else kv[1]["gap_points"])
        crit["promising: an LSH setting has a small discretization gap (read as: at most 5 points)"] = (
            v["gap_points"] <= 5, f"smallest: {e}, {v['gap_points']:.1f} points ({', '.join(f'{k} = {x}' for k, x in v['lsh_params'].items())})")
    L += ["", "## 4. Go / no-go (PLAN §9.2)", "", "| Criterion | Met? | Value |", "|---|---|---|"]
    for k, (ok, val) in crit.items():
        L.append(f"| {k} | {'yes' if ok else 'no'} | {val} |")
    best = (dv.get("best_encoder") or {}).get("0.01")
    cg = (dv.get("classifier_gap", {}).get(best["encoder"], {}) or {}).get("0.01") if best else None
    if cg:
        L += ["", f"- Classifier gap for the best encoder ({best['encoder']}): logistic regression {pct(cg['logreg_recall'])} "
              f"versus cosine {pct(cg['cosine_recall'])} ({cg['gap_points']:+.1f} points). A large positive gap points to the "
              "matching rule as the bottleneck; a small gap with low recall points to the encoder."]

    # ---- what the full run needs
    L += ["", "## 5. What the full run needs", ""]
    if len(c1) and best:
        p = min(max(best["recall_event"], 0.05), 0.95)
        need_ev = math.ceil(1.96 ** 2 * p * (1 - p) / 0.10 ** 2)
        L.append(f"- Event recall within +/-10 points at the observed recall ({pct(best['recall_event'])}) needs about {need_ev} "
                 f"validation events (binomial approximation); this run had {int(c1.n_events.max())}. The XD test set has 137 "
                 "fighting or abuse videos, of which the locked test keeps 40% of the source groups, so more events need the XD "
                 "training videos (weak labels) or a second violent source.")
        n_chk = int(c1.n_check.max())
        L.append(f"- FMR = 0.1% needs at least 3,000 benign val_check windows for 3 expected false matches, and about "
                 f"{math.ceil((1.96 / 0.5) ** 2 / 0.001):,} for a CI of about +/-50% of that value; this run had {n_chk:,}. "
                 "Windows within one SINS session are correlated (4 microphone nodes record the same moment), so the "
                 "effective number is smaller.")
        chk = sum(v["windows"] for k, v in cnt.items() if k.startswith("sins/val_check"))
        files, n_arch = facts.get("sins_files") or {}, len(meta["config"]["sins"]["archives"])
        if chk and files.get("used"):
            per_archive = chk / files["used"] * files["available"] / n_arch
            L.append(f"- One SINS archive holds about {files['available'] / n_arch:,.0f} ten-second files, which yield about "
                     f"{per_archive:,.0f} val_check windows with this split; about {math.ceil(15400 / per_archive)} of the 23 "
                     "archives give the +/-50% CI at FMR 0.1%.")
    lint = facts.get("report_lint")
    if lint:
        L += ["", "## 6. Check of comparison.md (PLAN §12.1)", "",
              f"- {'Passed' if lint['ok'] else 'Problems found'}: {lint['n_numbers']} numbers in {lint['n_lines']} bullets and "
              "table rows checked against facts.json; run ids, em-dashes, banned words, and section structure checked."]
        L += [f"- {p}" for p in lint["problems"][:30]]
    return "\n".join(L) + "\n"


# ============================================================================ comparison.md
def comparison_md(facts: dict) -> str:
    """The comparison-only report (PLAN §12.3), in Chenxinran's bullet style."""
    df, meta = _runs_df(facts), facts["meta"]
    dv = facts.get("derived") or derive(facts)
    L = ["# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching (pilot)", "",
         f"- Night {meta['night']}; plan version {meta['plan_version']}; code {meta['git']} (copy in {meta.get('code_copy', 'n/a')}). Every number below comes from "
         "`facts.json`, and each bullet names the runs it comes from."]
    if meta.get("smoke"):
        L.append("- This is a smoke test on a few videos and SINS files. Its numbers only show that the pipeline runs and "
                 "should not be read as results.")
    L.append("")
    if not len(df):
        return "\n".join(L + ["- No runs completed."]) + "\n"
    h = df[df.track == "home"]
    cos1 = h[(h.matcher == "cosine_top1") & (h.fmr_target == 0.01)].sort_values("recall_event", ascending=False)

    # ------------------------------------------------------------------ 1. datasets
    L += ["## 1. Datasets", ""]
    ds = facts.get("datasets") or {}
    S = {name: {r["fmr_target"]: r for r in rows} for name, rows in ds.get("settings", {}).items()}
    base = S.get("all_sources", {}).get(0.01)
    if base:
        L.append(f"- To compare datasets, we fixed the encoder ({ds['encoder']}, the encoder with the highest event recall in "
                 "Section 2) and the matcher (cosine similarity to the nearest reference, \"cosine top-1\"), and varied only "
                 "the source of the reference events, of the violent "
                 "test events, and of the benign windows that set and measure the false-match rate (FMR). Recall is counted "
                 "per validation event, and the area under the receiver operating characteristic curve (AUC) is computed "
                 "over windows.")
        L += ["", "| Setting | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | Runs |",
              "|---|---|---|---|---|---|"]
        for name, by in S.items():
            a, b = by.get(0.01), by.get(0.001)
            if a:
                L.append(f"| {a['description']} | {pct(a['recall_event'])} ({ci(a['recall_event_ci'])}) | "
                         f"{pct(b['recall_event']) if b else 'n/a'} | {pct(a['fmr_check'], 2)} | {f3(a['auc'])} | "
                         f"{a['run_id']}{', ' + b['run_id'] if b else ''} |")
        L.append(f"| SINS \"watching TV\" windows only (threshold from SINS) | n/a | n/a | {pct(base['tv_sins_fmr'], 2)} | n/a | "
                 f"{base['run_id']} |")
        snr = {s: int(re.search(r"[-+]\d+", s).group()) for s in base.get("tv_mix_by_snr") or {}}
        for s, v in (base.get("tv_mix_by_snr") or {}).items():
            L.append(f"| TV mix: violent audio mixed into SINS at {snr[s]} dB | n/a | n/a | {pct(v, 2)} | n/a | {base['run_id']} |")
        L.append("")
        mm, my = S.get("ref_movie__events_movie", {}).get(0.01), S.get("ref_movie__events_youtube", {}).get(0.01)
        if mm and my:
            lower = my["recall_event"] < mm["recall_event"] and my["recall_event_ci"][1] < mm["recall_event_ci"][0]
            L.append(f"- When the reference set contained only movie events, event recall at an FMR of 1% reached "
                     f"{pct(mm['recall_event'])} on movie events (95% CI {ci(mm['recall_event_ci'])}, n = {mm['n_events']}; run "
                     f"{mm['run_id']}), compared with {pct(my['recall_event'])} on YouTube events (95% CI "
                     f"{ci(my['recall_event_ci'])}, n = {my['n_events']}; run {my['run_id']}). "
                     + ("This suggests that matching may depend on the recording source of the reference clips as much as on "
                        "the incident itself." if lower else "The 95% CIs overlapped, so at this scale the source of the "
                        "reference clips did not appear to change recall clearly."))
        em, ey = S.get("events_movie", {}).get(0.01), S.get("events_youtube", {}).get(0.01)
        if em and ey:
            L.append(f"- With references from all sources, event recall at an FMR of 1% was {pct(em['recall_event'])} for movie "
                     f"events (run {em['run_id']}) and {pct(ey['recall_event'])} for YouTube events (run {ey['run_id']}).")
        cx, xs = S.get("calib_sins__check_xd_normal", {}).get(0.01), S.get("calib_xd_normal__check_sins", {}).get(0.01)
        if cx:
            moved = cx["fmr_check"] > 1.5 * 0.01 or cx["fmr_check"] < 0.01 / 1.5
            L.append(f"- A threshold calibrated on home recordings (SINS) for an FMR target of 1% produced a measured FMR of "
                     f"{pct(base['fmr_check'], 2)} on held-out SINS windows (run {base['run_id']}), compared with "
                     f"{pct(cx['fmr_check'], 2)} on windows from normal XD-Violence videos (run {cx['run_id']}). "
                     + ("Consequently, a threshold may not transfer between recording conditions, and benign calibration data "
                        "should potentially come from the deployment setting." if moved else
                        "The threshold therefore appeared to transfer to this second benign source."))
        if xs:
            L.append(f"- Additionally, a threshold calibrated on normal XD-Violence windows produced a measured FMR of "
                     f"{pct(xs['fmr_check'], 2)} on SINS windows and an event recall of {pct(xs['recall_event'])} (run "
                     f"{xs['run_id']}), compared with {pct(base['recall_event'])} under the SINS threshold (run {base['run_id']}).")
        tr = dv.get("tv_ratio", {}).get(ds["encoder"])
        if not _nan(base.get("tv_sins_fmr")):
            L.append(f"- On the benign side, the FMR on SINS \"watching TV\" windows reached {pct(base['tv_sins_fmr'], 2)} at the "
                     f"1% target, compared with {pct(base['fmr_check'], 2)} on all held-out SINS windows"
                     + (f", {tr['ratio']:.1f} times as high" if tr else "") + f" (run {base['run_id']}).")
        if base.get("tv_mix_by_snr"):
            parts = [f"{pct(v, 2)} at {snr[s]} dB" for s, v in base["tv_mix_by_snr"].items()]
            L.append(f"- Similarly, violent XD-Violence audio mixed into home recordings, as if played on a television, produced "
                     f"FMRs of {' and '.join(parts)} (run {base['run_id']}), compared with {pct(base['fmr_check'], 2)} on unmixed "
                     "SINS windows. We term these \"common-cause false matches\": benign windows that match because they "
                     "contain violent sound from another source.")
        by = base.get("sins_fmr_by_activity") or {}
        if by and max(by.values()) > 0:
            worst = [kv for kv in sorted(by.items(), key=lambda kv: -kv[1])[:3] if kv[1] > 0]
            L.append("- Among home activities, the highest FMRs at the 1% target occurred for "
                     + _enum([f"{k.replace('_', ' ')} ({pct(v, 2)})" for k, v in worst]) + f" (run {base['run_id']}).")
        for name, why in (ds.get("skipped") or {}).items():
            L.append(f"- The setting \"{name}\" was skipped ({why}).")
    else:
        L.append("- No dataset comparison completed.")

    # ------------------------------------------------------------------ 2. embeddings
    L += ["", "## 2. Embeddings", ""]
    enc_info = facts.get("encoders", {})
    if len(cos1):
        L.append("- To compare embeddings, we fixed the matcher (cosine top-1) and the threshold calibration (benign SINS "
                 "windows, FMR targets of 1% and 0.1%), and varied only the encoder.")
        L += ["", "| Encoder | Modality | Neural | Dimension | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) "
              "| AUC | Measured FMR (1% target) | Runs |", "|---|---|---|---|---|---|---|---|---|"]
        for r in cos1.itertuples():
            r01 = _row(h, encoder=r.encoder, matcher="cosine_top1", fmr_target=0.001)
            info = enc_info.get(r.encoder, {})
            L.append(f"| {r.encoder} | {info.get('modality', 'n/a')} | {'yes' if info.get('neural') else 'no'} | "
                     f"{info.get('dim', 'n/a')} | {pct(r.recall_event)} ({ci(r.recall_event_ci)}) | "
                     f"{pct(r01.recall_event) + ' (' + ci(r01.recall_event_ci) + ')' if r01 is not None else 'n/a'} | "
                     f"{f3(r.auc)} | {pct(r.fmr_check, 2)} | {r.run_id}{', ' + r01.run_id if r01 is not None else ''} |")
        L.append("")
        best, worst = cos1.iloc[0], cos1.iloc[-1]
        if len(cos1) > 1:
            L.append(f"- {best.encoder} reached the highest event recall at an FMR of 1% ({pct(best.recall_event)}, 95% CI "
                     f"{ci(best.recall_event_ci)}; run {best.run_id}), compared with {pct(worst.recall_event)} for "
                     f"{worst.encoder} (95% CI {ci(worst.recall_event_ci)}; run {worst.run_id}).")
        bg = dv.get("baseline_gap", {}).get(best.encoder, {}).get("0.01")
        if bg:
            L.append(f"- Compared with the non-neural baseline (MFCC statistics, {pct(bg['mfcc_recall'])}; run {bg['mfcc_run']}), "
                     f"{best.encoder} {'exceeded' if bg['gap_points'] >= 0 else 'fell below'} it by {abs(bg['gap_points']):.1f} "
                     f"points (run {bg['run']}). "
                     + ("Because the two 95% CIs did not overlap, this difference appears robust at this scale."
                        if not bg["ci_overlap"] else
                        "However, the two 95% CIs overlapped, so the difference may reflect sampling variation at this scale."))
        elif best.encoder == "mfcc_stats" and len(cos1) > 1:
            L.append(f"- The non-neural baseline (MFCC statistics) reached the highest recall (run {best.run_id}), which suggests "
                     "that the neural encoders did not add separation at this scale.")
        b01 = dv.get("best_encoder", {}).get("0.001")
        if b01:
            same = b01["encoder"] == best.encoder
            r_b = _row(h, encoder=best.encoder, matcher="cosine_top1", fmr_target=0.001)
            L.append(f"- At the stricter FMR target of 0.1%, {b01['encoder']} reached the highest event recall "
                     f"({pct(b01['recall_event'])}; run {b01['run_id']})"
                     + (", the same encoder as at 1%." if same or r_b is None else
                        f", whereas {best.encoder} reached {pct(r_b.recall_event)} (run {r_b.run_id})."))
        se, sp = cos1[cos1.encoder.isin(SOUND_EVENT)], cos1[cos1.encoder.isin(SPEECH)]
        if len(se) and len(sp):
            a, b = se.iloc[0], sp.iloc[0]
            ev = (f"{a.encoder} reached {pct(a.recall_event)} at an FMR of 1% (95% CI {ci(a.recall_event_ci)}; run {a.run_id}), "
                  f"compared with {pct(b.recall_event)} for {b.encoder} (95% CI {ci(b.recall_event_ci)}; run {b.run_id})")
            if a.recall_event_ci[0] > b.recall_event_ci[1]:
                L.append("- Sound-event encoders appear to separate violent and benign windows more reliably than the speech "
                         f"encoder. Specifically, {ev}.")
            elif b.recall_event_ci[0] > a.recall_event_ci[1]:
                L.append(f"- In contrast to our expectation, the speech encoder appears to separate violent and benign windows "
                         f"more reliably than the best sound-event encoder. Specifically, {ev}.")
            else:
                L.append(f"- The best sound-event encoder and the speech encoder did not differ clearly at this scale: {ev}, and "
                         "the 95% CIs overlapped.")
        cg = dv.get("classifier_gap", {}).get(best.encoder, {}).get("0.01")
        if cg:
            L.append(f"- A local logistic-regression classifier on the same embeddings, a non-private reference point, reached "
                     f"{pct(cg['logreg_recall'])} for {best.encoder} (run {cg['logreg_run']}), compared with "
                     f"{pct(cg['cosine_recall'])} for cosine matching (run {cg['cosine_run']}). We term this difference of "
                     f"{cg['gap_points']:.1f} points the \"classifier gap\". "
                     + ("A positive gap suggests that part of the loss may stem from matching against a reference list rather "
                        "than from the encoder itself." if cg["gap_points"] > 0 else
                        "Because the classifier did not exceed cosine matching, the encoder rather than the matching rule "
                        "appears to limit recall."))
    xd = df[(df.track == "xd") & (df.matcher == "cosine_top1") & (df.fmr_target == 0.01)]
    if len(xd):
        L += ["", "| Encoder (XD-Violence track: benign windows from normal XD videos) | Event recall at FMR 1% (95% CI) | AUC | "
              "Measured FMR | Run |", "|---|---|---|---|---|"]
        for r in xd.sort_values("recall_event", ascending=False).itertuples():
            L.append(f"| {r.encoder} | {pct(r.recall_event)} ({ci(r.recall_event_ci)}) | {f3(r.auc)} | {pct(r.fmr_check, 2)} | {r.run_id} |")
        L.append("")
        fused = xd[xd.encoder.str.startswith("fusion:")]
        if len(fused):
            f = fused.iloc[0]
            a_name, v_name = f.encoder[len("fusion:"):].split("+")
            a, v = _row(xd, encoder=a_name), _row(xd, encoder=v_name)
            if a is not None and v is not None:
                L.append(f"- Within XD-Violence, where benign windows came from normal XD videos, concatenating {a_name} with "
                         f"CLIP frame embeddings changed event recall at an FMR of 1% from {pct(a.recall_event)} with audio only "
                         f"(run {a.run_id}) and {pct(v.recall_event)} with video only (run {v.run_id}) to {pct(f.recall_event)} "
                         f"(run {f.run_id}), while the measured FMR was {pct(f.fmr_check, 2)}.")

    # ------------------------------------------------------------------ 3. matching
    L += ["", "## 3. Matching", ""]
    if len(cos1):
        be = cos1.iloc[0].encoder
        mt = h[(h.encoder == be) & (h.fmr_target == 0.01)]
        L.append(f"- To compare matching methods, we fixed the encoder ({be}) and the threshold calibration (benign SINS "
                 "windows), and varied only the matcher. For locality-sensitive hashing (LSH), k is the number of bits per "
                 "band, L the number of bands, and m the number of matching bands that calibration requires.")
        L += ["", "| Matcher | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | "
              "Threshold | Runs |", "|---|---|---|---|---|---|---|"]
        for r in mt.itertuples():
            r01 = _row(h, encoder=be, matcher_full=r.matcher_full, fmr_target=0.001)
            thr = f"m = {r.tau:g}" if r.matcher == "lsh" else "calibrated"
            L.append(f"| {r.matcher_full} | {pct(r.recall_event)} ({ci(r.recall_event_ci)}) | "
                     f"{pct(r01.recall_event) if r01 is not None else 'n/a'} | {pct(r.fmr_check, 2)} | {f3(r.auc)} | {thr} | "
                     f"{r.run_id}{', ' + r01.run_id if r01 is not None else ''} |")
        L.append("")
        g = dv.get("discretization_gap", {}).get(be, {}).get("0.01")
        if g:
            p = g["lsh_params"]
            L.append(f"- Converting embeddings into LSH codes for exact matching changed event recall from "
                     f"{pct(g['cosine_recall'])} with cosine similarity (run {g['cosine_run']}) to {pct(g['lsh_recall'])} with "
                     f"the best LSH setting (k = {p.get('k')}, L = {p.get('L')}, m = {g['m']:g}; run {g['lsh_run']}), a "
                     f"difference of {g['gap_points']:.1f} points that we term the \"discretization gap\".")
        lsh = mt[mt.matcher == "lsh"].sort_values("recall_event")
        if len(lsh) > 1:
            lo, hi = lsh.iloc[0], lsh.iloc[-1]
            L.append(f"- Across the LSH settings in the table, event recall at an FMR of 1% ranged from {pct(lo.recall_event)} "
                     f"({lo.params_s}; run {lo.run_id}) to {pct(hi.recall_event)} ({hi.params_s}; run {hi.run_id}).")
        c, hs = _row(mt, matcher="cosine_top1"), _row(mt, matcher="hamming_sign")
        if c is not None and hs is not None:
            between = g is not None and min(g["cosine_recall"], g["lsh_recall"]) <= hs.recall_event <= max(g["cosine_recall"], g["lsh_recall"])
            L.append(f"- Hamming distance on the sign bits of a principal component projection reached {pct(hs.recall_event)} "
                     f"(run {hs.run_id}), compared with {pct(c.recall_event)} for cosine similarity (run {c.run_id})"
                     + (", which lies between cosine similarity and LSH exact matching." if between else "."))
        t5, l2 = _row(mt, matcher="cosine_top5"), _row(mt, matcher="l2_raw")
        if c is not None and t5 is not None:
            L.append(f"- Averaging the similarity to the five nearest references (cosine top-5) reached {pct(t5.recall_event)} "
                     f"(run {t5.run_id}), compared with {pct(c.recall_event)} for the nearest reference alone (run {c.run_id}).")
        if c is not None and l2 is not None:
            L.append(f"- Similarly, the L2 distance to the nearest reference on centered, unnormalized embeddings reached "
                     f"{pct(l2.recall_event)} (run {l2.run_id}), compared with {pct(c.recall_event)} for cosine similarity "
                     f"(run {c.run_id}).")
    eq = facts.get("equivalence") or {}
    if eq:
        d, e = eq["n_disagree"], eq["n_disagree_explained_by_drops"]
        tail = ("" if not d else f" The remaining {d} windows were false negatives caused by band items dropped from the "
                "cuckoo table." if d == e else f" Of the {d} disagreements, {e} are explained by band items dropped from the "
                "cuckoo table.")
        L.append(f"- The Apple-style private set intersection (PSI) protocol reproduced the plaintext LSH decisions on "
                 f"{pct(eq['equivalence_rate'])} of {eq['n_windows']} windows, {eq['n_plain_positive']} of which matched in "
                 f"plaintext (k = {eq['k']}, L = {eq['L']}, m = {eq['m']}; runs {eq['lsh_run_id']} and {eq['run_id']})." + tail)
    if not len(cos1):
        L.append("- No matching comparison completed.")
    return "\n".join(L) + "\n"


# ============================================================================ report check (PLAN §12.1 step 3)
BANNED = re.compile(r"\b(prove|proves|conclusively|I|you)\b")
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")


def _known_numbers(facts: dict) -> set[str]:
    """Every numeric value in facts.json as it may be printed: as is, or as a percentage, with 0 to 3 decimals."""
    out: set[str] = set()

    def add(v: float):
        for x in (abs(v), abs(100 * v)):
            for d in range(4):
                out.add(f"{x:.{d}f}")
            out.add(f"{x:g}")

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            if not math.isnan(o) and not math.isinf(o):
                add(float(o))
        elif isinstance(o, str):
            if re.fullmatch(r"[-+]?\d+(\.\d+)?", o):
                add(float(o))
            for m in re.findall(r"[-+]?\d+", o):  # e.g. the SNR inside "tv_mix_-10dB"
                add(float(m))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk({k: v for k, v in facts.items() if k != "report_lint"})
    return out


def lint_report(md: str, facts: dict) -> dict:
    """Numbers traceable to facts.json, a run id on every bullet with a number, no em-dashes or banned words, and a
    table plus at least two bullets in each of the three sections."""
    known, problems, n_num, n_lines = _known_numbers(facts), [], 0, 0
    run_id = re.compile(re.escape(facts["meta"]["night"]) + r"-\d{4}")
    if "—" in md:
        problems.append("em-dash found")
    sections = re.split(r"^## ", md, flags=re.M)[1:]
    if len(sections) != 3:
        problems.append(f"expected 3 sections, found {len(sections)}")
    for sec in sections:
        lines = sec.splitlines()
        title, bullets = lines[0], [l for l in lines if l.startswith("- ")]
        rows = [l for l in lines if l.startswith("|") and not re.fullmatch(r"[|\-\s]+", l)]
        if not rows:
            problems.append(f"{title}: no table")
        if len(bullets) < 2:
            problems.append(f"{title}: fewer than two bullets")
        heads = {i for i, l in enumerate(lines) if l.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"[|\-\s]+", lines[i + 1])}
        first = next((i for i, l in enumerate(lines) if l.startswith("- ")), None)
        for i, line in enumerate(lines):
            if not (line.startswith("- ") or (line.startswith("|") and i not in heads and not re.fullmatch(r"[|\-\s]+", line))):
                continue
            n_lines += 1
            for w in BANNED.findall(line):
                problems.append(f"{title}: banned word \"{w}\" in: {line[:80]}")
            text = re.sub(r"top-\d+|Section \d", " ", run_id.sub(" ", line.replace("95% CI", "CI")))  # names, not numbers
            nums = [n.replace(",", "") for n in NUMBER.findall(text)]
            n_num += len(nums)
            for n in nums:
                if n not in known:
                    problems.append(f"{title}: {n} not found in facts.json: {line[:90]}")
            # the first bullet states what was held fixed (PLAN §12.2): settings, not results, so no run id
            if line.startswith("- ") and i != first and nums and not run_id.search(line) and "skipped" not in line:
                problems.append(f"{title}: bullet with numbers but no run id: {line[:90]}")
    return {"ok": not problems, "problems": problems, "n_numbers": n_num, "n_lines": n_lines}
