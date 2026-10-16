#!/usr/bin/env python3
"""Micro-benchmark: how cost scales with the server-set size |X| (pure Python => absolute numbers are
slow; the *shape* is the point).  Paper requirements being checked:

  * pdata grows linearly with |X|                                  (allowed: "proportional to the size of X")
  * client work and voucher size are independent of |X| and of m   (hard requirement of Section 2.3)
"""
import os
import time

from apple_psi.client import Client
from apple_psi.params import Params
from apple_psi.server import Server

params = Params(t=5, s_max=10, ad_len=64)
print(f"{'|X|':>7} {'setup s':>8} {'pdata KB':>9} {'voucher B':>10} {'client ms':>10} {'server ms':>10}")
for n in (100, 400, 1600):
    X = [os.urandom(12) for _ in range(n)]
    server = Server(params)
    t0 = time.perf_counter(); pdata = server.setup(X); setup = time.perf_counter() - t0
    client = Client(params, pdata)
    reps = 30
    vouchers, t0 = [], time.perf_counter()
    for i in range(reps):
        vouchers.append(client.create_real_voucher(X[i] if i % 2 else os.urandom(12), os.urandom(64), os.urandom(16)).to_bytes())
    c = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    for v in vouchers:
        server.ingest_bytes(v)
    s = (time.perf_counter() - t0) / reps * 1000
    print(f"{n:>7} {setup:>8.2f} {len(pdata.to_bytes()) / 1024:>9.1f} {len(vouchers[0]):>10} {c:>10.1f} {s:>10.1f}")
