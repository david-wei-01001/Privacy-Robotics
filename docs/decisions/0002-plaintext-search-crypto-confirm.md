# 0002: Search in plaintext, confirm in crypto
- Status: accepted (2026-10-26)
- Context: the design space is large, and the private protocols are slow.
- Decision: explore with plaintext matchers on cached embeddings. Run only the finalists through the private
  protocols, which must reproduce the decisions of their plaintext twin (PLAN §6).
- Consequences: an equivalence test for every private adapter.
