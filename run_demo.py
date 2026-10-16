#!/usr/bin/env python3
"""End-to-end demonstration of the Apple PSI (ftPSI-AD) prototype.

    python run_demo.py                                   # Fashion-MNIST, 12 matches, threshold 5 -> data revealed
    python run_demo.py --matches 4                       # below threshold -> server learns ids only
    python run_demo.py --dataset digits                  # offline (needs scikit-learn)
    python run_demo.py --dataset synthetic               # random 96-bit hashes, no images
    python run_demo.py --dataset folder --folder PATH    # your own images
    python run_demo.py --perturb                         # client holds edited copies of watch-list images

Roles
-----
  server  holds X, a set of 96-bit perceptual hashes (a toy "watch list") -- published only as pdata.
  client  holds a photo library; for every photo it streams one fixed-size voucher to the server.
          Associated data = an 8x8 thumbnail ("low-resolution visual derivative").
  The script checks the server's output against the ideal functionality F of the paper (Figure 1).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

from apple_psi import datasets as ds
from apple_psi.client import Client
from apple_psi.params import Params
from apple_psi.server import Server


def build_pool(args):
    """-> (hashes, images-or-None). Distinct perceptual hashes, each with a representative image."""
    if args.dataset == "synthetic":
        return ds.synthetic_hashes(args.server_size + args.client_size + 10), None
    if args.dataset == "fashion-mnist":
        images = ds.load_fashion_mnist("t10k")
    elif args.dataset == "digits":
        images = ds.load_digits()
    elif args.dataset == "folder":
        if not args.folder:
            sys.exit("--folder is required with --dataset folder")
        images = ds.load_image_folder(args.folder)
    else:
        raise AssertionError(args.dataset)
    hashes, imgs = ds.unique_by_hash(images)
    return hashes, imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["fashion-mnist", "digits", "synthetic", "folder"], default="fashion-mnist")
    ap.add_argument("--folder")
    ap.add_argument("--server-size", type=int, default=300, help="|X|: hashes in the server set")
    ap.add_argument("--client-size", type=int, default=60, help="m: photos in the client library")
    ap.add_argument("--matches", type=int, default=12, help="how many client photos are drawn from the server set")
    ap.add_argument("--threshold", "-t", type=int, default=5, help="t: data revealed iff #real matches > t")
    ap.add_argument("--s-max", type=int, default=10, help="upper bound on synthetic vouchers")
    ap.add_argument("--synthetic-prob", type=float, default=0.1, help="chance that a new photo is designated synthetic")
    ap.add_argument("--hashes", type=int, default=2, choices=[2, 3], help="number of cuckoo hash functions")
    ap.add_argument("--expansion", type=float, default=None, help="cuckoo table size / |X| (default 2.5, or 1.4 with 3 hashes)")
    ap.add_argument("--perturb", action="store_true", help="matching photos are edited copies (hash may or may not survive)")
    ap.add_argument("--seed", type=int, default=0, help="seed for *demo data selection* only (crypto uses os.urandom)")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    if args.matches > args.client_size:
        sys.exit("--matches cannot exceed --client-size")

    # ---------------------------------------------------------------- data
    print(f"== data: {args.dataset}")
    pool_hashes, pool_imgs = build_pool(args)
    need = args.server_size + args.client_size - args.matches
    if len(pool_hashes) < need:
        sys.exit(f"dataset has only {len(pool_hashes)} distinct hashes, need {need}")
    order = list(range(len(pool_hashes)))
    rnd.shuffle(order)
    server_idx = order[:args.server_size]
    other_idx = order[args.server_size:args.server_size + args.client_size - args.matches]
    X = [pool_hashes[i] for i in server_idx]

    # client library: (hash, associated data, label) per photo
    library = []
    for i in rnd.sample(server_idx, args.matches):
        if pool_imgs is None:
            y, ad = pool_hashes[i], f"photo-{i}".encode().ljust(ds.AD_LEN, b"\0")
        else:
            img = ds.perturb(pool_imgs[i], nprng) if args.perturb else pool_imgs[i]
            y, ad = ds.phash96(img), ds.visual_derivative(img)
        library.append((y, ad))
    for i in other_idx:
        if pool_imgs is None:
            library.append((pool_hashes[i], f"photo-{i}".encode().ljust(ds.AD_LEN, b"\0")))
        else:
            library.append((pool_hashes[i], ds.visual_derivative(pool_imgs[i])))
    rnd.shuffle(library)
    print(f"   server set |X| = {len(X)} hashes of {ds.HASH_LEN * 8} bits; client library m = {len(library)} photos")

    # ---------------------------------------------------------------- server setup
    params = Params(t=args.threshold, s_max=args.s_max, ad_len=ds.AD_LEN, num_hashes=args.hashes,
                    table_expansion=args.expansion or (2.5 if args.hashes == 2 else 1.4))
    print(f"\n== server setup  (t={params.t}, s_max={params.s_max}, {params.num_hashes} cuckoo hashes)")
    server = Server(params)
    t0 = time.perf_counter()
    pdata = server.setup(X)
    blob = pdata.to_bytes()
    print(f"   pdata: {len(blob):,} bytes  ({pdata.table_size} table cells; {len(server.dropped)} set element(s) "
          f"did not fit -> false negatives)   [{time.perf_counter() - t0:.2f}s]")

    # ---------------------------------------------------------------- client streams vouchers
    print("\n== client streams one voucher per photo; server screens each on arrival")
    client = Client(params, blob, synthetic_prob=args.synthetic_prob)
    ids = []
    c_time = s_time = 0.0
    sizes = set()
    first_reveal = None
    for k, (y, ad) in enumerate(library, 1):
        id_ = os.urandom(16)
        ids.append(id_)
        t0 = time.perf_counter()
        voucher = client.create_voucher(y, ad, id_)
        wire = voucher.to_bytes()
        c_time += time.perf_counter() - t0
        sizes.add(len(wire))
        t0 = time.perf_counter()
        status = server.ingest_bytes(wire)
        s_time += time.perf_counter() - t0
        if status == "match" and first_reveal is None and server.reveal().above_threshold:
            first_reveal = k
    print(f"   {len(library)} vouchers, every one exactly {sizes.pop() if len(sizes) == 1 else sorted(sizes)} bytes")
    print(f"   client: {c_time / len(library) * 1000:.1f} ms/voucher   server: {s_time / len(library) * 1000:.1f} ms/voucher")
    print(f"   server screening: {server.stats['non-match']} non-match (discarded), "
          f"{server.stats['match']} candidate match, {server.stats['invalid']} invalid")
    if first_reveal:
        print(f"   threshold crossed when voucher #{first_reveal} arrived")

    # ---------------------------------------------------------------- server output vs ideal functionality
    print("\n== server output")
    out = server.reveal()
    X_in_table = set(X) - set(server.dropped)
    synthetic = set(client.state.synthetic_ids)
    expected_real = {i: ad for i, (y, ad) in zip(ids, library) if y in X_in_table and i not in synthetic}
    print(f"   ground truth: {len(expected_real)} real matches (> t={params.t}? {len(expected_real) > params.t}), "
          f"{len(synthetic)} synthetic vouchers designated by the client")
    print(f"   server learned all {len(out.idlist)} voucher ids (public) and {len(out.outset_ids)} 'match' ids")
    if out.above_threshold:
        print(f"   ABOVE THRESHOLD: associated data decrypted for {len(out.revealed)} photos; "
              f"{len(out.synthetic_ids)} ids identified as synthetic")
        ok = out.revealed == expected_real and out.synthetic_ids == synthetic
        if pool_imgs is not None:
            for id_, ad in list(out.revealed.items())[:3]:
                print(f"\n   revealed visual derivative of voucher {id_.hex()[:8]}...:")
                print("      " + ds.ascii_thumbnail(ad).replace("\n", "\n      "))
    else:
        print("   BELOW THRESHOLD: no associated data decrypted; real and synthetic matches are indistinguishable")
        ok = (not out.revealed) and out.outset_ids == set(expected_real) | synthetic
        for note in out.notes:
            print("   note:", note)
    print(f"\n== result matches the ideal functionality F of the paper: {'YES' if ok else 'NO  <-- BUG'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
