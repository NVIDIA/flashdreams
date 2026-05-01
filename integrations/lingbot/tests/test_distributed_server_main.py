from __future__ import annotations

from argparse import Namespace

from lingbot.webrtc import server


class _FakeSessionManager:
    def __init__(self) -> None:
        self.wait_called = False
        self.exit_called = False

    def wait_for_termination(self) -> None:
        self.wait_called = True

    def send_exit_signal(self) -> None:
        self.exit_called = True


def _args() -> Namespace:
    return Namespace(
        host="127.0.0.1",
        port=8080,
        config_name="LingBot-World-Fast",
        no_compile=False,
        device="cuda:0",
    )


def test_main_rank0_sends_exit_signal(monkeypatch) -> None:
    fake_manager = _FakeSessionManager()

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server, "parse_args", _args)
    monkeypatch.setattr(
        server,
        "LingbotWebRTCSessionManager",
        lambda runtime_config: fake_manager,
    )
    monkeypatch.setattr(server, "create_app", lambda session_manager: object())
    monkeypatch.setattr(server.web, "run_app", lambda app, host, port: None)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server.dist, "is_initialized", lambda: False)

    server.main()

    assert fake_manager.exit_called is True
    assert fake_manager.wait_called is False


def test_main_worker_rank_waits_for_termination(monkeypatch) -> None:
    fake_manager = _FakeSessionManager()

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(server, "parse_args", _args)
    monkeypatch.setattr(server, "distributed_init", lambda: 1)
    monkeypatch.setattr(server.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(server.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        server,
        "LingbotWebRTCSessionManager",
        lambda runtime_config: fake_manager,
    )

    server.main()

    assert fake_manager.wait_called is True
    assert fake_manager.exit_called is False
