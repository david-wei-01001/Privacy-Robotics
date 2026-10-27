# 0006: Repo and data layout
- Status: accepted (2026-10-27); item 3 superseded by 0007 (2026-10-27)
- Decisions:
  1) Work on `main`; Elise keeps backups.
  2) Remove the duplicate flat copy of `apple_psi` at the repo root.
  3) Keep data, caches, models, and the environment in `~/research-data/psi-robotics/`, outside the
     iCloud-synced Desktop. (Superseded by 0007: `data/` and `.venv/` inside the repo, in a `.nosync` folder.)
  4) XD-Violence: use Elise's download (moved into `raw/xd_violence/`), with the Hugging Face mirror as the
     fallback.
  5) Decode with PyAV instead of a system ffmpeg.
- Evidence: iCloud symlink for Desktop; `~/Downloads` not readable by the agent; diff of the root files.
