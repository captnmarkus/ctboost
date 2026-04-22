import socket
import time

import numpy as np
from sklearn.datasets import make_classification


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_tcp_listener(port: int, *, host: str = "127.0.0.1", timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            detail = "" if last_error is None else f": {last_error}"
            raise TimeoutError(
                f"timed out waiting for tcp listener at {host}:{port}{detail}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(min(max(remaining, 0.05), 0.5))
            try:
                sock.connect((host, port))
                return
            except OSError as exc:
                last_error = exc
        time.sleep(0.05)


def make_classification_data():
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    return X.astype(np.float32), y.astype(np.float32)
