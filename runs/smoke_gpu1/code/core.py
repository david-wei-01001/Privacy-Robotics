"""Core helpers: registry (Strategy pattern, PLAN §2), paths, logging, heartbeat, stage timing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------- registry
_REG: dict[str, dict[str, type]] = {}


def register(kind: str, name: str):
    def deco(cls):
        _REG.setdefault(kind, {})[name] = cls
        cls.registry_name = name
        return cls
    return deco


def build(kind: str, spec: dict | str):
    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    return _REG[kind][spec.pop("name")](**spec)


def available(kind: str) -> list[str]:
    return sorted(_REG.get(kind, {}))


# ----------------------------------------------------------------------------- paths
def data_root() -> Path:
    """`data_root` from paths.yaml; a relative path is resolved against the repo root (PLAN §13)."""
    cfg = yaml.safe_load((REPO / "paths.yaml").read_text())
    p = Path(os.path.expanduser(cfg["data_root"]))
    return p if p.is_absolute() else REPO / p


# Model weights are cached under <data_root>/models/ (PLAN §13.2); a value already set in the environment wins.
os.environ.setdefault("HF_HOME", str(data_root() / "models/hf"))
os.environ.setdefault("TORCH_HOME", str(data_root() / "models/torch"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops missing on Apple's MPS backend run on the CPU


def plan_version() -> str:
    import re
    m = re.search(r"\*\*Plan version (\d+\.\d+)", (REPO / "docs/plan/PLAN.md").read_text(encoding="utf-8"))
    return m.group(1) if m else "unknown"


def git_state() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- logging
def get_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("mmpsi")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(log_path), logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


class Heartbeat:
    """Writes the current stage to a file every `period` seconds (PLAN §9.2)."""

    def __init__(self, path: Path, period: int = 60):
        self.path, self.period, self.stage, self.t0 = path, period, "start", time.time()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, stage: str):
        self.stage = stage
        self._write()

    def _write(self):
        self.path.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": self.stage,
                                         "elapsed_min": round((time.time() - self.t0) / 60, 1)}))

    def _run(self):
        while not self._stop.wait(self.period):
            self._write()

    def stop(self):
        self._stop.set()


class StageLog:
    """Records status, duration and errors of each stage for feasibility.md."""

    def __init__(self, path: Path):
        self.path = path
        self.stages: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def start(self, name: str):
        self.stages[name] = {"status": "running", "start": time.time()}
        self._save()

    def done(self, name: str, status: str = "ok", **info):
        s = self.stages.setdefault(name, {"start": time.time()})
        s.update(status=status, seconds=round(time.time() - s["start"], 1), **info)
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.stages, indent=2, default=str))
