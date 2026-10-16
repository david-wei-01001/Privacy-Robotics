# Apple PSI (ftPSI-AD) — a Python prototype

An educational, from-scratch implementation of the private-set-intersection protocol in
*"The Apple PSI System"* (Bhowmick, Boneh, Myers, Talwar, Tarbe — July 2021): **fuzzy threshold PSI with
associated data (ftPSI-AD)**. Pure Python 3, ~1,300 lines of library code (including docs), 50 tests.

> **Not for production.** The elliptic-curve code is pure Python and not constant-time, there is no
> authenticated channel, and several system-level pieces of the paper are out of scope (see
> [Limitations](#limitations)). It exists to make the protocol runnable, inspectable and testable.

## Does the paper come with a dataset or code?

No to both. The paper is a protocol specification and security analysis; I found no code released by Apple.
The protocol is *data-agnostic*: it only ever sees opaque byte strings (the hashes `y`) and fixed-length
associated data `ad`. In Apple's deployment the inputs are 96-bit NeuralHash values of photos, and the
server's set comes from a hash list supplied by child-safety organisations that is **not public** — and
that this project neither needs nor attempts to recreate.

So this repo uses benign stand-ins with the same *shape* (`apple_psi/datasets.py`):

| source | what it is | needs |
|---|---|---|
| `fashion-mnist` (default) | 10,000 28×28 images (Zalando, MIT licence), downloaded from GitHub on first run | internet once |
| `digits` | scikit-learn's bundled 1,797 8×8 images | `scikit-learn`, offline |
| `synthetic` | uniformly random 96-bit strings | nothing |
| `folder` | any directory of images you supply | – |

Images are mapped to **96-bit perceptual hashes** (`phash96`: 32×32 grayscale → DCT → 96 low-frequency
coefficients vs. their median — same output size as NeuralHash, but a classic non-neural hash). The
associated data is an **8×8 thumbnail** (64 bytes), standing in for Apple's "visual derivative". Once the
threshold is exceeded the server can decrypt these thumbnails — the demo prints them as ASCII art.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                      # 50 tests, ~10 s
python run_demo.py                       # 12 matches, t=5  -> server decrypts the matching thumbnails
python run_demo.py --matches 5           # exactly t       -> server learns nothing but ids
python run_demo.py --matches 4 --synthetic-prob 0.15   # synthetic vouchers blur the real count
python run_demo.py --dataset digits | synthetic | folder --folder PATH
python run_demo.py --perturb             # client holds edited copies of watch-list images
python run_demo.py --hashes 3            # 3 cuckoo hash functions instead of 2
python benchmark.py                      # cost vs. |X|
```

Library use:

```python
from apple_psi import Params, Server, Client

params = Params(t=5, s_max=10, ad_len=64)        # threshold, max synthetics, associated-data length
server = Server(params)
pdata  = server.setup(X)                         # X: iterable[bytes]; publish pdata.to_bytes() to all clients

client = Client(params, pdata.to_bytes(), synthetic_prob=0.1)
voucher = client.create_voucher(y, ad)           # per triple, immediately on arrival; len == params.voucher_len
server.ingest_bytes(voucher.to_bytes())          # server screens it right away ("non-match" / "match")

out = server.reveal()                            # periodically
out.above_threshold, out.revealed, out.synthetic_ids, out.outset_ids
```

Sample output (`python run_demo.py`):

```
== server setup  (t=5, s_max=10, 2 cuckoo hashes)
   pdata: 24,841 bytes  (750 table cells; 0 set element(s) did not fit -> false negatives)
== client streams one voucher per photo; server screens each on arrival
   60 vouchers, every one exactly 412 bytes
   server screening: 44 non-match (discarded), 16 candidate match, 0 invalid
== server output
   ground truth: 12 real matches (> t=5? True), 4 synthetic vouchers designated by the client
   ABOVE THRESHOLD: associated data decrypted for 12 photos; 4 ids identified as synthetic
== result matches the ideal functionality F of the paper: YES
```

## Map from paper to code

| paper | file |
|---|---|
| §2 problem statement, §2.3 constraints (message pattern, equal-length vouchers) | `messages.py`, `params.py` |
| §3 `(Enc,Dec)` AES-128-GCM, `H0` HKDF, PRF `F` (HMAC) | `crypto_utils.py` |
| §3 NIST P-256, random-oracle `H` (RFC 9380 `P256_XMD:SHA-256_SSWU_RO_`) | `ec.py`, `hash_to_curve.py` |
| §3 Shamir sharing over `F_Sh` | `shamir.py` |
| §3.1 Naor–Reingold DH random self-reduction | `client.py` (`create_real_voucher`), tested in `test_primitives.py` |
| §3.2, §4.3 `(s,t)`-detectable hash function + Coppersmith–Sudan-style detection | `dhf.py` |
| §4.1 server setup: cuckoo table, `L = αG`, `P_i = α·H(T[i])` or random | `server.py` (`setup`), `cuckoo.py` |
| §4.2 client voucher, Case 1 (real) / Case 2 (synthetic) | `client.py` |
| §4.2 server processing: `Ŝ=αQ`, decrypt `ct`→`rkey`→`rct`, detect, reconstruct `adkey` | `server.py` (`ingest`, `reveal`) |
| §4.4 ideal functionality `F` (Figure 1) | asserted in `tests/test_protocol.py` and `run_demo.py` |
| §5 multi-device accounts (shared `ClientState`, share x-coordinate = `F(fkey,id)`) | `client.py`, `test_protocol.py` |
| §2.3 "Updates" (old and new vouchers count together) | `Server.setup(keep_key=True)`, `test_pdata_update_…` |

### How the pieces fit (one voucher)

1. Client encrypts the associated data under its account key: `adct = Enc(adkey, ad)`.
2. It derives a Shamir share `(x, f(x))` of `adkey`, with `x = F(fkey, id)`, and `r = DHF(hkey, x0)`.
3. It picks `rkey`, and sets `rct = Enc(rkey, (r, adct, share))`.
4. For each cuckoo cell `w = h_j(y)`, it runs the Naor–Reingold reduction on `(L, H(y), P_w)`, giving
   `(Q_j, S_j)`, and stores `ct_j = Enc(H0(S_j), rkey)`. If `y ∈ X` then `S_j = α·Q_j` for exactly one `j`.
5. Server computes `α·Q_j`, tries to open each `ct_j`; if one opens it learns `rkey` and opens `rct`.
   A non-member `y` yields an independent random key, so nothing opens.
6. Once more than `t` distinct shares have opened, the DHF detection separates real shares from
   synthetic ones, `adkey` is reconstructed, and the `adct` of the real matches are decrypted.

## Design decisions and deviations (be aware)

* **Parameters.** `t` = threshold (the Decentralized Thoughts blog post says Apple's announced value was 30; the demo uses 5
  so it runs in seconds). The server learns data iff **more than** `t` real matches exist, i.e. `t+1` shares
  reconstruct `adkey` (degree-`t` polynomial). `F_ell = GF(2^64−59)`, `F_Sh = GF(2^130−5)`.
* **Cuckoo table size.** Two hash functions need load < 0.5, so the default is `n' = 2.5·n`, which places
  essentially everything (items that don't fit are dropped and become false negatives, Remark 3; the demo
  prints how many). `--hashes 3` uses 3 positions (`n' = 1.4·n`), at the cost of one more `(Q, ct)` pair per
  voucher. Remark 7 (XOR-style secret-shared cells giving ~1.25·n and a single point per voucher) is **not**
  implemented.
* **Synthetic policy.** The paper leaves the distribution of synthetic matches to Apple's technical summary.
  Here each new triple is designated synthetic with probability `synthetic_prob`, hard-capped at `s_max`.
  A designated id stays synthetic for its lifetime.
* **Detection** is the basic scheme of §4.3: it requires ≥ `t+1` real and ≤ `s_max` random columns. With
  more than `s_max` synthetics it can fail or mis-classify (Remark 8); the extended Coppersmith–Sudan
  decoder of Remark 9 is not implemented.
* **De-duplication.** The server keeps one record per voucher `id` and also treats shares with the same
  `x` as one, so a triple re-sent from several devices counts once. Different ids with identical hashes
  (copies of the same image) count separately, exactly as in §5 — the paper handles that *outside* the protocol.
* **Hash-to-curve** follows the final RFC 9380 rather than the draft cited in the paper, and is checked
  against the RFC's test vectors.
* The voucher `id` is public by design; every voucher has the same length (412 bytes with the demo
  parameters) whether it is a match, a non-match or synthetic.

## Limitations

Out of scope: the secure authenticated channel; third-party certification of `pdata` / "overfitting" defence
(Remark 5); distribution of `pdata` via OS updates; the real perceptual hash (NeuralHash); everything
outside the cryptographic protocol (human review, account-level policy, abuse prevention).
Security caveats of this code: variable-time big-integer arithmetic, Python's non-zeroizing memory, and no
review by cryptographers. The proofs in the paper (and Bellare's game-based analysis) cover the *protocol*,
not this implementation.

Exact-match PSI is only as good as the hash feeding it. With `--perturb`, only the edited copies whose
`phash96` is bit-identical to the original are detected; the rest are simply missed (the demo's ground truth
accounts for this). A hash that is robust to edits is a separate problem from the PSI.

## Verification performed

* `ec.py` scalar multiplication cross-checked against OpenSSL (via `cryptography`); group order and
  point-encoding edge cases tested.
* `hash_to_curve.py` reproduces the RFC 9380 test vectors for `P256_XMD:SHA-256_SSWU_RO_`.
* DH self-reduction property (§3.1) tested for both DH and non-DH tuples.
* DHF detection tested on randomised mixtures: exact recovery with ≥ `t+1` real / ≤ `s` random columns, and
  failure (hiding) with exactly `t` real columns.
* End-to-end tests of every regime of the ideal functionality (none / `= t` / `> t` real matches, with and
  without synthetics, at the exact `t` vs `t+1` boundary), multi-device duplicates, pdata updates, malformed
  vouchers/pdata, 3-hash cuckoo, and cuckoo drops (false negatives only — never false positives).
* Mutation checks: deliberately breaking the protocol (wrong cuckoo cell, missing `γ·L`, off-by-one
  threshold, no DHF filtering, non-DH synthetic, variable-length voucher, share x not derived from `id`)
  makes the suite fail.

## Performance (pure Python, one core, `python benchmark.py`)

| `\|X\|` | setup | pdata | voucher | client / voucher | server / voucher |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.6 s | 8 KB | 412 B | 11 ms | 3.6 ms |
| 400 | 2.3 s | 32 KB | 412 B | 11 ms | 3.6 ms |
| 1,600 | 8.9 s | 129 KB | 412 B | 11 ms | 3.7 ms |

pdata and setup are linear in `|X|`; client work and voucher size do not depend on `|X|` (a hard
requirement in §2.3).
