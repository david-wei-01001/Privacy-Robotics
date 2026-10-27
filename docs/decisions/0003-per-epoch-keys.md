# 0003: Fresh keys and threshold per (robot, epoch)
- Status: accepted (2026-10-26)
- Context: Apple's threshold accumulates over an account's lifetime, and `s_max` caps synthetic vouchers per
  client lifetime.
- Evidence: at 0.1 % per-window FMR with 8,640 windows a day, t = 30 is crossed by benign false matches in
  about 3.6 days. A continuous stream would use up `s_max` within minutes.
- Decision: a fresh `ClientState` per (robot, epoch); server sessions keyed by (robot, epoch) (PLAN §3.6).
