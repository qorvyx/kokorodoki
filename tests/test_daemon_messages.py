import socket
import time

import pytest

from utils import is_client_message, is_command_message, read_client_message


def test_read_client_message_reads_until_close():
    server, client = socket.socketpair()
    with server, client:
        client.sendall(b"TEXT:hello")
        client.close()
        assert read_client_message(server, timeout=1) == b"TEXT:hello"


def test_read_client_message_times_out_on_silent_client():
    server, client = socket.socketpair()
    with server, client:
        start = time.monotonic()
        assert read_client_message(server, timeout=0.2) is None
        assert time.monotonic() - start < 1


def test_read_client_message_times_out_when_client_never_closes():
    server, client = socket.socketpair()
    with server, client:
        client.sendall(b"\x00\x00\x00\x0chost:version")
        assert read_client_message(server, timeout=0.2) is None


@pytest.mark.parametrize(
    "data",
    [b"TEXT:hello", b"IMAGE:\x89PNG\r\n\x1a\n\x00", b"!pause", b"!speed 1.2"],
)
def test_is_client_message_accepts_client_payloads(data):
    assert is_client_message(data)


@pytest.mark.parametrize(
    "data", [b"", b"   ", b"\x00\x01\x02\x03binary", b"\xff\xfe\xfd", b"CNXN\x00\x00"]
)
def test_is_client_message_rejects_junk(data):
    assert not is_client_message(data)


@pytest.mark.parametrize("data", [b"!pause", b"!speed 1.2", b"!exit"])
def test_bare_commands_are_commands(data):
    assert is_command_message(data)


@pytest.mark.parametrize(
    "data", [b"TEXT:!important note", b"TEXT:!exit now", b"TEXT:hello", b"hello"]
)
def test_text_payloads_are_never_commands(data):
    assert not is_command_message(data)
