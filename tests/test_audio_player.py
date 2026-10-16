import sys
import threading
import types

import numpy as np
import pytest

from models import AudioPlayer


class FakeStream:
    def __init__(self, **kwargs):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sd(monkeypatch):
    sd = types.ModuleType("sounddevice")
    sd._initialized = 0
    sd.streams = []

    def _initialize():
        sd._initialized += 1

    def _terminate():
        sd._initialized -= 1

    def output_stream(**kwargs):
        stream = FakeStream(**kwargs)
        sd.streams.append(stream)
        return stream

    sd._initialize = _initialize
    sd._terminate = _terminate
    sd.OutputStream = output_stream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def wait_for(condition, timeout=2.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if condition():
            return True
        done.wait(0.01)
    return condition()


def test_no_stream_until_play(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    assert player.stream is None
    assert fake_sd.streams == []
    assert fake_sd._initialized == 0


def test_idle_close_releases_portaudio(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.play(np.zeros(10))
    assert fake_sd._initialized == 1
    assert fake_sd.streams[0].started

    player.stop()
    player.schedule_close()
    assert wait_for(lambda: player.stream is None)
    assert fake_sd.streams[0].closed
    assert fake_sd._initialized == 0

    # The next clip reopens the stream and reinitializes PortAudio.
    player.play(np.zeros(10))
    assert len(fake_sd.streams) == 2
    assert fake_sd._initialized == 1
    player.close()
    assert fake_sd._initialized == 0


def test_play_within_grace_period_keeps_stream(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.1)
    player.play(np.zeros(10))
    player.stop()
    player.schedule_close()
    player.play(np.zeros(10))

    threading.Event().wait(0.3)
    assert player.stream is fake_sd.streams[0]
    assert not fake_sd.streams[0].closed
    assert len(fake_sd.streams) == 1
    player.close()


def test_no_close_while_audio_pending(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.play(np.zeros(10))
    player.pause()
    player.schedule_close()

    threading.Event().wait(0.2)
    assert player.stream is not None
    assert fake_sd._initialized == 1
    player.close()
