"""Educational Python prototype of the Apple PSI system (ftPSI-AD).

Reference: A. Bhowmick, D. Boneh, S. Myers, K. Talwar, K. Tarbe, "The Apple PSI System", July 2021.

    from apple_psi import Params, Server, Client

    params = Params(t=5, s_max=10, ad_len=64)
    server = Server(params);  pdata = server.setup(X)                  # X: iterable of bytes (hashes)
    client = Client(params, pdata.to_bytes(), synthetic_prob=0.1)
    server.ingest_bytes(client.create_voucher(y, ad).to_bytes())      # once per triple, as it arrives
    out = server.reveal()                                              # ServerOutput

NOT FOR PRODUCTION USE: pure-Python, non-constant-time arithmetic; no authenticated channel.
"""
from .client import Client, ClientState
from .messages import PData, Voucher
from .params import Params
from .server import Server, ServerOutput

__all__ = ["Params", "Server", "ServerOutput", "Client", "ClientState", "PData", "Voucher"]
__version__ = "0.1.0"
