import socket

import pytest


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    """CLI tests start consoles; never open a browser window while testing."""
    monkeypatch.setenv("MACBOT_NO_BROWSER", "1")


@pytest.fixture
def busy_port():
    """Hold a port without sending requests or depending on shared machine ports."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]
