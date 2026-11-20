from types import SimpleNamespace

import pytest

from macbot import web
from macbot.cli import main
from macbot.config import ConsolePortError, load


@pytest.mark.parametrize(("section", "default"), [("init", 8765), ("teleop", 8766), ("run", 8767), ("replay", 8768)])
def test_busy_default_falls_forward_but_explicit_ports_do_not(tmp_path, busy_port, section, default):
    cfg = load()
    assert cfg[section]["port"] == default
    # Substitute a controlled default so the test does not bind fixed ports on a shared machine.
    cfg[section]["port"] = busy_port
    with web.bind_console(cfg, section, web.Handler) as server:
        assert busy_port < server.server_port < busy_port + web.PORT_ATTEMPTS
        assert server.url == f"http://127.0.0.1:{server.server_port}"
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}: {{port: {busy_port}}}\n")
    with pytest.raises(ConsolePortError, match=rf"Port {busy_port} .*--port"):
        web.bind_console(load(path), section, web.Handler)
    with pytest.raises(ConsolePortError, match=rf"Port {busy_port} .*--port"):
        web.bind_console(cfg, section, web.Handler, busy_port)
    # Explicit zero means let the OS choose, not use the configured port.
    with web.bind_console(load(path), section, web.Handler, 0) as server:
        assert server.server_port not in {0, busy_port}


def test_equal_to_builtin_port_is_still_explicit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("teleop: {port: 8766}\n")
    assert load(path).explicit_ports == {"teleop"}


def test_init_passes_actual_url_to_browser_without_opening_it(busy_port, monkeypatch, capsys, tmp_path):
    from macbot.web import init_wizard

    cfg = load()
    cfg["init"]["port"] = busy_port
    monkeypatch.setattr(init_wizard, "load", lambda _: cfg)
    opened = []

    def capture(url):
        opened.append(url)
        raise KeyboardInterrupt

    monkeypatch.setattr(init_wizard.web, "open_browser", capture)
    setup = SimpleNamespace(path=tmp_path / "config.yaml", refresh=lambda: None)
    assert init_wizard.serve(setup) is None
    url, = opened
    assert int(url.rsplit(":", 1)[-1]) > busy_port
    assert f"Setup: {url}" in capsys.readouterr().out


@pytest.mark.parametrize(("command", "args"), [
    ("init", ["--no-browser"]), ("collect", ["--fake"]), ("run", ["--fake", "--policy", "unused"]), ("replay", []),
])
def test_cli_port_collision_is_one_line(tmp_path, busy_port, command, args):
    path = tmp_path / "config.yaml"
    path.write_text(f"data_root: {tmp_path}\n")
    if command == "replay":
        args = [*args, "--dir", str(tmp_path)]
    with pytest.raises(SystemExit, match=rf"\APort {busy_port} [^\n]+--port[^\n]+\Z"):
        main(["--config", str(path), command, "--port", str(busy_port), *args])


def test_fallback_skips_the_other_consoles_defaults(busy_port):
    cfg = load()
    cfg["teleop"]["port"] = busy_port
    cfg["run"]["port"] = busy_port + 1
    cfg["replay"]["port"] = busy_port + 2
    with web.bind_console(cfg, "teleop", web.Handler) as server:
        assert server.server_port not in {busy_port, busy_port + 1, busy_port + 2}


def test_open_browser_hands_url_to_the_sudo_user(monkeypatch):
    import macbot.web as web

    calls = []
    monkeypatch.delenv("MACBOT_NO_BROWSER")
    monkeypatch.setattr(web.sys, "platform", "darwin")
    monkeypatch.setattr(web.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "501")
    monkeypatch.setenv("SUDO_USER", "alice")
    monkeypatch.setattr(web.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(web.webbrowser, "open", lambda url: calls.append(["webbrowser", url]))
    assert web.open_browser("http://127.0.0.1:8769")
    assert calls == [["launchctl", "asuser", "501", "sudo", "-u", "alice", "/usr/bin/open", "http://127.0.0.1:8769"]]

    calls.clear()
    monkeypatch.setattr(web.os, "geteuid", lambda: 501)
    web.open_browser("http://127.0.0.1:8766")
    assert calls == [["webbrowser", "http://127.0.0.1:8766"]]

    calls.clear()
    monkeypatch.setenv("MACBOT_NO_BROWSER", "1")
    assert not web.open_browser("http://127.0.0.1:8766") and calls == []
