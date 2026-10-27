# 0001: Modular pipeline built on the Strategy pattern
- Status: accepted (2026-10-26)
- Context: dataset, encoder, similarity, and metric are all uncertain, and should be swappable without
  editing the pipeline.
- Options: 1) a single script per experiment; 2) interfaces (`typing.Protocol`) with interchangeable
  strategies, selected by a registry and configs.
- Decision: option 2 (PLAN §2, §3).
- Consequences: a new encoder or matcher is a new file. Runs are described entirely by their config.
