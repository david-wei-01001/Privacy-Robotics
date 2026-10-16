"""End-to-end tests of the ftPSI-AD protocol (paper Section 4.2) against the ideal functionality
of Figure 1:

    if |id(Y ∩ X) \\ S| > t :  server learns  Y[id(Y ∩ X) \\ S]_{id,ad}  and  S
    else                      :  server learns  id(Y ∩ X) ∪ S   (and no associated data)
"""
import os

import pytest

from apple_psi.client import Client, ClientState
from apple_psi.messages import PData, Voucher
from apple_psi.params import Params
from apple_psi.server import Server

PARAMS = Params(t=3, s_max=5, ad_len=16)


def ad_for(i: int) -> bytes:
    return f"image-{i:04d}".encode().ljust(PARAMS.ad_len, b".")


@pytest.fixture()
def setting():
    X = [os.urandom(12) for _ in range(40)]
    for _ in range(25):            # 2-hash cuckoo insertion fails with probability O(1/n): re-roll the
        server = Server(PARAMS)    # hash functions so that tests about *protocol* logic are deterministic
        pdata = server.setup(X)
        if not server.dropped:
            break
    assert not server.dropped
    return X, server, pdata


def run(server, client, real_hits, non_hits=0, synthetics=0):
    """Feed the server `real_hits` matching triples, `non_hits` non-matching ones and
    `synthetics` synthetic ones.  Returns (ids of real hits -> ad, set of synthetic ids, ids of non-hits)."""
    hits, synth, others = {}, set(), set()
    for y in real_hits:
        id_ = os.urandom(16)
        ad = ad_for(len(hits))
        hits[id_] = ad
        server.ingest(client.create_real_voucher(y, ad, id_))
    for _ in range(non_hits):
        id_ = os.urandom(16)
        others.add(id_)
        server.ingest(client.create_real_voucher(os.urandom(12), ad_for(999), id_))
    for _ in range(synthetics):
        v = client.create_synthetic_voucher()
        synth.add(v.id)
        server.ingest(v)
    return hits, synth, others


# ------------------------------------------------------------------ tPSI-AD (no synthetics)
def test_no_matches_server_learns_only_ids(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    _, _, others = run(server, client, [], non_hits=8)
    out = server.reveal()
    assert set(out.idlist) == others and out.outset_ids == set()
    assert not out.above_threshold and out.revealed == {}
    assert server.stats["non-match"] == 8


def test_at_threshold_ids_only(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, _, _ = run(server, client, X[:PARAMS.t], non_hits=5)       # exactly t matches
    out = server.reveal()
    assert out.outset_ids == set(hits)                                # ids of the matches are learnt...
    assert not out.above_threshold and out.revealed == {}             # ...but no associated data


def test_above_threshold_reveals_exactly_the_matches(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, _, others = run(server, client, X[:PARAMS.t + 1], non_hits=6)   # t+1 matches
    out = server.reveal()
    assert out.above_threshold
    assert out.revealed == hits                                       # all ad of matches, nothing else
    assert out.synthetic_ids == set()
    assert not (set(out.revealed) & others)


def test_many_matches(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, _, _ = run(server, client, X[:25], non_hits=10)
    out = server.reveal()
    assert out.revealed == hits


def test_streaming_reveal_is_monotone(setting):
    """Calling reveal() repeatedly while vouchers stream in: nothing before t+1 matches, all after."""
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    for i, y in enumerate(X[:PARAMS.t + 3]):
        server.ingest(client.create_real_voucher(y, ad_for(i), os.urandom(16)))
        out = server.reveal()
        assert out.above_threshold == (i + 1 > PARAMS.t)
        assert len(out.revealed) == (i + 1 if out.above_threshold else 0)


# ------------------------------------------------------------------ ftPSI-AD (with synthetics)
def test_synthetics_hide_real_matches_below_threshold(setting):
    """2 real + 4 synthetic: the server sees 6 > t 'matches' but cannot tell which are real."""
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, synth, _ = run(server, client, X[:2], non_hits=3, synthetics=4)
    out = server.reveal()
    assert out.outset_ids == set(hits) | synth                         # fuzzy intersection
    assert not out.above_threshold and out.revealed == {}


def test_synthetics_are_filtered_above_threshold(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, synth, _ = run(server, client, X[:7], non_hits=4, synthetics=PARAMS.s_max)
    out = server.reveal()
    assert out.above_threshold
    assert out.revealed == hits                                        # synthetic ids carry no ad
    assert out.synthetic_ids == synth                                  # and are identified as S


def test_threshold_boundary_with_max_synthetics(setting):
    X, server, pdata = setting
    # exactly t real + s_max synthetic -> hidden
    c1 = Client(PARAMS, pdata)
    run(server, c1, X[:PARAMS.t], synthetics=PARAMS.s_max)
    assert not server.reveal().above_threshold
    # one more real match -> everything real is revealed
    server.ingest(c1.create_real_voucher(X[PARAMS.t], ad_for(77), os.urandom(16)))
    out = server.reveal()
    assert out.above_threshold and len(out.revealed) == PARAMS.t + 1


def test_synthetic_policy_respects_s_max(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata, synthetic_prob=1.0)                 # designate everything synthetic...
    ids = [os.urandom(16) for _ in range(PARAMS.s_max + 4)]
    for id_ in ids:
        client.create_voucher(X[0], ad_for(0), id_)
    assert len(client.state.synthetic_ids) == PARAMS.s_max             # ...but never more than s_max
    designated = set(client.state.synthetic_ids)
    for id_ in ids:                                                    # designation is stable per id
        assert (id_ in designated) == (id_ in client.state.synthetic_ids)
        client.create_voucher(X[0], ad_for(0), id_)
    assert client.state.synthetic_ids == designated


# ------------------------------------------------------------------ wire format
def test_all_vouchers_have_identical_length(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    real = client.create_real_voucher(X[0], ad_for(0), os.urandom(16))
    miss = client.create_real_voucher(os.urandom(12), ad_for(1), os.urandom(16))
    synth = client.create_synthetic_voucher()
    lengths = {len(v.to_bytes()) for v in (real, miss, synth)}
    assert lengths == {PARAMS.voucher_len}
    for v in (real, miss, synth):
        assert Voucher.from_bytes(v.to_bytes(), PARAMS).to_bytes() == v.to_bytes()


def test_server_status_codes_via_bytes(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    assert server.ingest_bytes(client.create_real_voucher(X[0], ad_for(0), os.urandom(16)).to_bytes()) == "match"
    assert server.ingest_bytes(client.create_real_voucher(os.urandom(12), ad_for(0), os.urandom(16)).to_bytes()) == "non-match"
    assert server.ingest_bytes(client.create_synthetic_voucher().to_bytes()) == "match"


def test_malformed_vouchers_are_rejected(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    good = client.create_real_voucher(X[0], ad_for(0), os.urandom(16)).to_bytes()
    with pytest.raises(ValueError):
        Voucher.from_bytes(good[:-1], PARAMS)                          # wrong length
    corrupted = bytearray(good)
    corrupted[16] = 0x05                                               # invalid point prefix
    with pytest.raises(ValueError):
        Voucher.from_bytes(bytes(corrupted), PARAMS)
    tampered = bytearray(good)
    tampered[-1] ^= 1                                                  # break the rct tag
    assert server.ingest_bytes(bytes(tampered)) == "non-match"         # decrypts to nothing


def test_associated_data_must_have_fixed_length(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    with pytest.raises(ValueError):
        client.create_real_voucher(X[0], b"too short", os.urandom(16))


# ------------------------------------------------------------------ pdata
def test_pdata_roundtrip_and_size(setting):
    X, server, pdata = setting
    blob = pdata.to_bytes()
    again = PData.from_bytes(blob)
    assert again.to_bytes() == blob and again.table_size == pdata.table_size
    assert pdata.table_size == int(PARAMS.table_expansion * len(X))
    assert len(blob) == 10 + 16 * 3 + 33 * (1 + pdata.table_size)


def test_client_rejects_bad_pdata(setting):
    X, server, pdata = setting
    dup = PData(pdata.L, [pdata.points[0]] + pdata.points[:-1], pdata.hash_nonce, pdata.cuckoo_nonces)
    with pytest.raises(ValueError):
        Client(PARAMS, dup)
    with pytest.raises(ValueError):
        PData.from_bytes(pdata.to_bytes()[:-1])
    with pytest.raises(ValueError):
        Client(Params(t=3, s_max=5, ad_len=16, num_hashes=3), pdata)   # parameter mismatch


def test_pdata_points_are_distinct_valid_points(setting):
    """Member cells (alpha*H(x)) and empty cells (random) are all distinct, valid points."""
    X, server, pdata = setting
    assert len({(p.x, p.y) for p in pdata.points}) == pdata.table_size
    assert all(not p.is_infinity() for p in pdata.points)
    n_member_cells = sum(1 for _ in X)
    assert pdata.table_size > n_member_cells


# ------------------------------------------------------------------ multi-device (Section 5)
def test_duplicate_triples_across_devices_count_once(setting):
    X, server, pdata = setting
    dev1 = Client(PARAMS, pdata)
    dev2 = Client(PARAMS, pdata, state=ClientState.from_json(dev1.state.to_json()))   # same account
    ids = [os.urandom(16) for _ in range(PARAMS.t)]                    # t distinct matching triples
    for i, id_ in enumerate(ids):
        server.ingest(dev1.create_real_voucher(X[i], ad_for(i), id_))
        server.ingest(dev2.create_real_voucher(X[i], ad_for(i), id_))  # same triple on the second device
    out = server.reveal()
    assert len(server.idlist) == 2 * PARAMS.t
    assert not out.above_threshold and out.outset_ids == set(ids)      # still only t distinct shares
    # one genuinely new match pushes the count to t+1 and everything opens
    dev2_id = os.urandom(16)
    server.ingest(dev2.create_real_voucher(X[PARAMS.t], ad_for(9), dev2_id))
    out = server.reveal()
    assert out.above_threshold and set(out.revealed) == set(ids) | {dev2_id}


def test_same_id_always_yields_the_same_share_and_dhf_value(setting):
    """Section 5: the Shamir x-coordinate (and the DHF input x0) are PRF(fkey, id), so re-sending a
    triple -- from this device or another one of the same account -- reproduces the identical share."""
    X, server, pdata = setting
    dev1 = Client(PARAMS, pdata)
    dev2 = Client(PARAMS, pdata, state=ClientState.from_json(dev1.state.to_json()))
    id_ = os.urandom(16)
    records = []
    for dev in (dev1, dev2, dev1):
        server.ingest(dev.create_real_voucher(X[0], ad_for(0), id_))
        rec = server._matches[id_]
        records.append((rec.x, rec.y, rec.r))
    assert records[0] == records[1] == records[2]
    # ...whereas a different id gives a different share
    other = os.urandom(16)
    server.ingest(dev1.create_real_voucher(X[0], ad_for(0), other))
    rec = server._matches[other]
    assert (rec.x, rec.y, rec.r) != records[0]


def test_pdata_update_keeps_old_vouchers_compatible():
    """Section 2.3 'Updates': vouchers made before and after a pdata update jointly count to the threshold."""
    X_old = [os.urandom(12) for _ in range(30)]
    X_new = [os.urandom(12) for _ in range(30)]                     # a completely different server set
    server = Server(PARAMS)
    old_pdata = server.setup(X_old)
    client = Client(PARAMS, old_pdata)
    hits = {}
    for i, y in enumerate([x for x in X_old if x not in set(server.dropped)][:2]):
        id_ = os.urandom(16)
        hits[id_] = ad_for(i)
        server.ingest(client.create_real_voucher(y, hits[id_], id_))
    assert not server.reveal().above_threshold
    new_pdata = server.setup(X_new, keep_key=True)                   # OS update ships new pdata
    client_after_update = Client(PARAMS, new_pdata, state=client.state)
    for i, y in enumerate([x for x in X_new if x not in set(server.dropped)][:2]):
        id_ = os.urandom(16)
        hits[id_] = ad_for(10 + i)
        server.ingest(client_after_update.create_real_voucher(y, hits[id_], id_))
    out = server.reveal()                                             # 2 + 2 = 4 > t = 3
    assert out.above_threshold and out.revealed == hits


def test_state_json_roundtrip():
    st = ClientState.generate(PARAMS)
    st.synthetic_ids.add(os.urandom(16))
    again = ClientState.from_json(st.to_json())
    assert again == st


def test_independent_accounts_do_not_combine(setting):
    """Shares from two different accounts (different adkey) must not open each other's data."""
    X, server, pdata = setting
    a, b = Client(PARAMS, pdata), Client(PARAMS, pdata)
    for i in range(PARAMS.t):                                          # t matches each => 2t distinct shares
        server.ingest(a.create_real_voucher(X[i], ad_for(i), os.urandom(16)))
        server.ingest(b.create_real_voucher(X[i], ad_for(i), os.urandom(16)))
    out = server.reveal()
    assert not out.above_threshold and out.revealed == {}


# ------------------------------------------------------------------ parameter variants
def test_three_cuckoo_hash_functions():
    params = Params(t=3, s_max=4, ad_len=16, num_hashes=3, table_expansion=1.4)
    X = [os.urandom(12) for _ in range(60)]
    server = Server(params)
    pdata = server.setup(X)
    client = Client(params, pdata.to_bytes())
    present = [x for x in X if x not in set(server.dropped)]
    hits = {}
    for i, y in enumerate(present[:6]):
        id_ = os.urandom(16)
        hits[id_] = ad_for(i)
        server.ingest(client.create_real_voucher(y, hits[id_], id_))
    for _ in range(3):
        server.ingest(client.create_synthetic_voucher())
    out = server.reveal()
    assert out.above_threshold and out.revealed == hits and len(out.synthetic_ids) == 3
    assert params.voucher_len == 16 + 3 * (33 + 44) + params.rct_len


def test_dropped_cuckoo_items_are_false_negatives_only():
    params = Params(t=3, s_max=4, ad_len=16, table_expansion=1.15)     # deliberately over-full
    X = [os.urandom(12) for _ in range(60)]
    server = Server(params)
    pdata = server.setup(X)
    assert server.dropped, "expected some items to be dropped at this load"
    client = Client(params, pdata)
    dropped = set(server.dropped)
    for x in X:
        status = server.ingest(client.create_real_voucher(x, ad_for(0), os.urandom(16)))
        assert (status == "non-match") == (x in dropped)               # never a false positive


def test_server_set_with_duplicates_and_single_element():
    server = Server(PARAMS)
    y = os.urandom(12)
    pdata = server.setup([y, y, y])
    client = Client(PARAMS, pdata)
    assert server.ingest(client.create_real_voucher(y, ad_for(0), os.urandom(16))) == "match"
    with pytest.raises(ValueError):
        Server(PARAMS).setup([])


def test_detection_tolerates_more_than_t_plus_one_real_and_exact_synthetic_count(setting):
    X, server, pdata = setting
    client = Client(PARAMS, pdata)
    hits, synth, _ = run(server, client, X[:30], non_hits=5, synthetics=PARAMS.s_max)
    out = server.reveal()
    assert out.revealed == hits and out.synthetic_ids == synth
