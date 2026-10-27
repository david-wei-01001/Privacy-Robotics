# Privacy-Robotics: multimodal PSI for privacy-preserving incident reporting

- **Start here:** [INSTRUCTIONS.md](INSTRUCTIONS.md): setup, datasets, embeddings, similarity methods, and the
  Night 1 results
- Environment: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`
- Plan (living document, versioned): `docs/plan/PLAN.md`; history: `docs/plan/CHANGELOG.md`
- Decisions: `docs/decisions/`; morning reviews: `docs/reviews/`
- Apple PSI prototype (crypto core): `apple_psi_prototype/` (see its README)
- Multimodal pipeline: `mmpsi/`; configs: `configs/`; runs and reports: `runs/`, `reports/`
- Large data, caches, model weights, and logs live in `data/` (the data root set in `paths.yaml`), and the Python
  environment in `.venv/`; git ignores both. The project folder sits under `Research-Projects.nosync`, which
  iCloud does not sync (PLAN §13, ADR 0007).

Overnight pilot: `bash tools/night.sh <night_id>` (see PLAN §9.2). Smoke test first:
`.venv/bin/python -m mmpsi.pilot --night smoke1 --smoke`.
