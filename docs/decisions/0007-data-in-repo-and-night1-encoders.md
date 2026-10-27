# 0007: Data and environment inside the repo; Night-1 encoders
- Status: accepted (2026-10-27); supersedes item 3 of 0006.
- Context: the project moved to `~/Desktop/Research-Projects.nosync/`, which iCloud does not sync, so the reasons
  for keeping data outside the repo (iCloud uploads, "Optimize Mac Storage" evicting files mid-run) no longer apply.
  The old environment in `~/research-data/psi-robotics/venv` installs `apple_psi` in editable mode from a path that
  no longer exists. Elise downloaded the official XD-Violence VGGish and I3D feature archives. PLAN §9.2 listed
  encoders that `mmpsi/features.py` does not implement (PANNs CNN14, BEATs, I3D).
- Options: 1) keep `~/research-data/psi-robotics/` as the data root; 2) keep data, caches, models, and logs in
  `data/` and the environment in `.venv/`, both inside the repo and git-ignored. For the encoders: a) implement
  PANNs, BEATs, and I3D before Night 1; b) run the encoders the code implements and move the others to Night 2.
- Decisions:
  1) Option 2. `paths.yaml` sets `data_root: data`, resolved against the repo root; `HF_HOME` and `TORCH_HOME`
     default to `data/models/`; `.gitignore` lists `data/`, `.venv/`, and `*.egg-info/`.
  2) The uv environment (Python 3.12) is `.venv/` at the repo root, with `apple_psi_prototype` installed in editable
     mode.
  3) The official feature archives stay zipped in `data/raw/xd_violence/official/`; the pilot never extracts them
     (`data.find_local_videos` skips `official/`).
  4) Option b. Night 1 runs `mfcc_stats`, `vggish`, `hubert` (layer 9), `ast`, `clap`, and `clip_frames`
     (`configs/pilot_n1.yaml`). PANNs and BEATs need extra packages and checkpoints hosted outside Hugging Face; I3D
     needs the 0.64 s snippets of the official features aligned to the 4.8 s windows, read from the zip without
     extracting it. All three move to Night 2.
- Evidence: `apple_psi_prototype` tests in `.venv`: 50 passed (2026-10-27); the smoke run `smoke1` completed every
  stage (`reports/smoke1/feasibility.md`); `git check-ignore` confirms that `data/` and `.venv/` are ignored; both
  official archives pass a zip check (`data/raw/xd_violence/SOURCE.md`).
- Consequences: one folder holds code and data, and git sees only small files. The old environment
  (`~/research-data/psi-robotics/venv`, 1.2 GB) is unused; deleting it is Elise's call.
