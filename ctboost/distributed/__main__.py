"""CLI entrypoint for `python -m ctboost.distributed`."""

from __future__ import annotations

import sys

from .server import run_distributed_collective_server


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m ctboost.distributed <host> <port>")
    run_distributed_collective_server(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    main()
