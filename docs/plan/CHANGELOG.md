# Plan changelog

Newest first. Each entry records the version, the time, the sections that changed, and three lines written by
whoever made the change: Trigger (who or what asked), Why, and Evidence. `tools/plan_log.py new` writes the
mechanical part. Entries 0.1 to 0.5 were reconstructed from the planning conversation; they were never committed.

## v0.8 (2026-10-27 03:56)
- Sections: changed: 9.2 Night 1: overnight pilot (small scale, laptop only), 12.2 Writing rules (her voice, adapted to bullets), 13.4 Version history so far (+5 / -3 lines)
- Trigger: Night 1 (2026-10-27_n1) results: three encoders at or near 100% event recall at FMR 1% on the home track
- Why: A bootstrap CI over events collapses to one point when every event is detected, so CI-overlap statements and the go/no-go CI width overstated certainty; comparisons where both options reach 100% at FMR 1% carry no information
- Evidence: clap detected 67 of 67 events at FMR 1% (run 2026-10-27_n1-0023): bootstrap CI 100.0 to 100.0, exact Clopper-Pearson 94.6 to 100.0; reports/2026-10-27_n1/ re-rendered with --report-only, lint passes
- Decisions: 0008

## v0.7 (2026-10-27 02:24)
- Sections: changed: 3.2 Repository layout (proposed), 9.2 Night 1: overnight pilot (small scale, laptop only), 13.1 Three places, three purposes, 13.2 Layout, 13.4 Version history so far (+52 / -37 lines)
- Trigger: Elise moved the project to ~/Desktop/Research-Projects.nosync and the data from ~/research-data into data/; her handoff of 2026-10-27 asked to align the plan with the new layout and with the code.
- Why: iCloud does not sync .nosync folders, so data, caches, models, and logs live in data/ and the environment in .venv/, both git-ignored; the old venv installed apple_psi from a path that no longer exists. The official XD VGGish and I3D feature archives are recorded in §9.2 and §13.2. §9.2 now lists the encoders the code implements (PANNs, BEATs, and I3D move to Night 2) and states what the pilot runs: TV mix at -10 and 0 dB as in configs/pilot_n1.yaml (the plan said -15 dB), m set by calibration (the plan said m in {1, 2}), and 150 PSI windows, half of them violent (the plan said about 200).
- Evidence: apple_psi_prototype tests in .venv: 50 passed; smoke run smoke1 completed every stage (reports/smoke1/feasibility.md); git check-ignore confirms data/ and .venv/; annotations.txt sha256 unchanged after the move (data/raw/xd_violence/SOURCE.md).
- Decisions: 0007

## v0.6 (2026-10-27 00:45)
- Sections: header; §9.2 (needed from Elise, items 1 and 2); §13.1; §13.4
- Trigger: Elise said no separate branch is needed and the original code need not stay unchanged (she keeps
  backups); she is downloading XD-Violence herself.
- Why: work directly on `main`. Remove the duplicate flat copy of `apple_psi` at the repo root, which would
  shadow library modules such as `datasets`. XD files go to `<data_root>/raw/xd_violence/`, because the agent
  cannot read `~/Downloads`. Decode audio and video with PyAV inside the environment instead of installing
  ffmpeg system-wide.
- Evidence: `ls ~/Downloads` fails with "Operation not permitted" (2026-10-27); the root files were
  byte-identical to `apple_psi_prototype/` (diff, 2026-10-27).
- Decisions: 0006

## v0.5 (2026-10-27 00:35)
- Sections: §13 (new): storage layout and change tracking; §3.2; §3.5; §9.2 item 6
- Trigger: Elise asked how files will be stored, and how later plan iterations will be recorded automatically.
- Why: the Desktop is synced to iCloud, so data, caches, and the environment move to
  `~/research-data/psi-robotics/`. One versioned plan, plus a changelog, decision records, and run records
  that point back to the plan version.
- Evidence: `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop -> ~/Desktop` (2026-10-27).
- Decisions: 0006

## v0.4 (2026-10-27 00:30)
- Sections: §0 item 6; §9 (rewritten); §4.2; §4.7
- Trigger: Elise asked why the milestones span weeks; she wants one night that runs a small-scale version.
- Why: replace the weekly milestones with an overnight pilot (Night 1), then nightly cycles. XD-Violence
  comes from the Hugging Face mirror, because the official OneDrive links need a Microsoft login.
- Evidence: SharePoint links redirect to login.microsoftonline.com; `jherng/xd-violence` serves 800 test
  videos (10.9 GB) without login; label codes confirmed in the mirror's loader (2026-10-27).
- Decisions: 0005

## v0.3 (2026-10-27)
- Sections: §12 (rewritten)
- Trigger: Elise: the report should only contain comparisons of datasets, embeddings, and matching.
- Why: §12 is narrowed to three comparison sections. Leakage, cost, and protocol results stay in
  `results.tsv` and `facts.json`.
- Decisions: 0004

## v0.2 (2026-10-27)
- Sections: §12 (new); §0; §1; §3.2; §8.2; §9
- Trigger: Elise: output a report written as bullet points in her writing style.
- Why: a frozen report generator plus a style- and number-checked bullet report.
- Decisions: 0004

## v0.1 (2026-10-26)
- Sections: all (first version)
- Trigger: Elise asked for a plan for a modular autoresearch pipeline for multimodal PSI.
- Why: a Strategy-pattern pipeline; dataset, embedding, and matching modules; evaluation; an autoresearch
  loop; weekly milestones.
- Decisions: 0001, 0002, 0003
