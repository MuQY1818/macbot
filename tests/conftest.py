import socket

import pytest


@pytest.fixture
def busy_port():
    """Hold a port without sending requests or depending on shared machine ports."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]
