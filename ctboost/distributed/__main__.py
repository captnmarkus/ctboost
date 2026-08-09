"""CLI entrypoint for `python -m ctboost.distributed`."""

from __future__ import annotations

import os
import sys

from .server import run_distributed_collective_server
from .tcp import MAX_PAYLOAD_BYTES


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m ctboost.distributed <host> <port>")
    auth_token = os.environ.get("CTBOOST_DISTRIBUTED_AUTH_TOKEN", "")
    run_id = os.environ.get("CTBOOST_DISTRIBUTED_RUN_ID", "")
    raw_world_size = os.environ.get("CTBOOST_DISTRIBUTED_WORLD_SIZE", "")
    if not auth_token or not run_id or not raw_world_size:
        raise SystemExit("authenticated distributed coordinator configuration is missing")
    run_distributed_collective_server(
        sys.argv[1],
        int(sys.argv[2]),
        auth_token=auth_token,
        expected_run_id=run_id,
        expected_world_size=int(raw_world_size),
        max_payload_bytes=int(
            os.environ.get("CTBOOST_DISTRIBUTED_MAX_PAYLOAD_BYTES", MAX_PAYLOAD_BYTES)
        ),
        request_timeout=float(
            os.environ.get("CTBOOST_DISTRIBUTED_REQUEST_TIMEOUT", "30")
        ),
        collective_timeout=float(
            os.environ.get("CTBOOST_DISTRIBUTED_COLLECTIVE_TIMEOUT", "600")
        ),
    )


if __name__ == "__main__":
    main()
