import socket

import numpy as np
from sklearn.datasets import make_classification


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_classification_data():
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    return X.astype(np.float32), y.astype(np.float32)
