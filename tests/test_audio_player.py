import sys
import threading
import types

import numpy as np
import pytest

import models
from models import AudioPlayer, TTSPlayer


class FakeStream:
    fail_start = False

    def __init__(self, **kwargs):
        self.started = False
        self.closed = False
        self.callback = kwargs["callback"]

    def start(self):
        if FakeStream.fail_start:
            raise RuntimeError("device busy")
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
    monkeypatch.setattr(models, "TERMINATE_PORTAUDIO_ON_IDLE", True)
    monkeypatch.setattr(FakeStream, "fail_start", False)
    return sd


def wait_for(condition, timeout=2.0):
    pause = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if condition():
            return True
        pause.wait(0.01)
    return condition()


def test_no_stream_until_play(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    assert player.stream is None
    assert fake_sd.streams == []
    assert fake_sd._initialized == 0


def test_idle_close_releases_portaudio(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.acquire()
    player.play(np.zeros(10))
    assert fake_sd._initialized == 1
    assert fake_sd.streams[0].started

    player.stop()
    player.release()
    assert wait_for(lambda: player.stream is None)
    assert fake_sd.streams[0].closed
    assert fake_sd._initialized == 0

    # The next clip reopens the stream and reinitializes PortAudio.
    player.acquire()
    player.play(np.zeros(10))
    assert len(fake_sd.streams) == 2
    assert fake_sd._initialized == 1
    player.close()
    assert fake_sd._initialized == 0


def test_new_session_within_grace_period_keeps_stream(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.1)
    player.acquire()
    player.play(np.zeros(10))
    player.stop()
    player.release()
    player.acquire()

    threading.Event().wait(0.3)
    assert player.stream is fake_sd.streams[0]
    assert not fake_sd.streams[0].closed
    player.close()


def test_open_session_blocks_close_from_older_session(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.acquire()  # old playback
    player.acquire()  # new playback, waiting between clips
    player.play(np.zeros(10))
    player.stop()
    player.release()  # old playback ends

    threading.Event().wait(0.2)
    assert player.stream is not None
    player.release()
    assert wait_for(lambda: player.stream is None)


def test_idle_close_drops_clip_left_loaded(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.acquire()
    player.play(np.zeros(10))
    player.pause()
    player.release()  # session ended on an error path with the clip loaded

    assert wait_for(lambda: player.stream is None)
    assert not player.is_playing
    assert fake_sd._initialized == 0


def test_failed_start_is_not_cached(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    FakeStream.fail_start = True
    with pytest.raises(RuntimeError):
        player.play(np.zeros(10))
    assert player.stream is None
    assert fake_sd.streams[0].closed
    assert fake_sd._initialized == 0

    FakeStream.fail_start = False
    player.play(np.zeros(10))
    assert fake_sd.streams[1].started
    player.close()


def test_portaudio_kept_when_terminate_disabled(fake_sd, monkeypatch):
    monkeypatch.setattr(models, "TERMINATE_PORTAUDIO_ON_IDLE", False)
    player = AudioPlayer(24000, idle_delay=0.05)
    player.play(np.zeros(10))
    player.close()
    assert fake_sd.streams[0].closed
    assert fake_sd._initialized == 1


def test_blocking_play_returns_when_clip_ends(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    done = threading.Event()

    def play():
        player.play(np.zeros(10), blocking=True)
        done.set()

    threading.Thread(target=play, daemon=True).start()
    assert wait_for(lambda: player.stream is not None)
    outdata = np.zeros((64, 2))
    player.stream.callback(outdata, 64, None, None)
    assert done.wait(1)
    player.close()


def make_tts_player():
    return TTSPlayer(None, "a", "af_heart", 1.0, False)


def test_pause_and_resume_before_first_playback():
    player = make_tts_player()
    player.pause_playback()
    player.resume_playback()


def test_change_language_to_current_keeps_pipeline():
    pipeline = object()
    player = make_tts_player()
    player.pipeline = pipeline
    assert player.change_language("a", None)
    assert player.pipeline is pipeline


def test_pause_holds_the_next_clip(fake_sd):
    player = AudioPlayer(24000, idle_delay=0.05)
    player.play(np.ones(10))
    player.pause()
    player.stop()
    player.play(np.ones(10))
    outdata = np.full((4, 2), 9.0)
    player.stream.callback(outdata, 4, None, None)
    assert not outdata.any()
    player.resume()
    player.stream.callback(outdata, 4, None, None)
    assert outdata.all()
    player.close()


def test_failed_language_load_keeps_current_language(monkeypatch):
    def broken_pipeline(**kwargs):
        raise RuntimeError("missing japanese extra")

    monkeypatch.setattr(models, "KPipeline", broken_pipeline)
    pipeline = types.SimpleNamespace(model=object())
    player = make_tts_player()
    player.pipeline = pipeline
    with pytest.raises(RuntimeError):
        player.change_language("j", None)
    assert player.language == "a"
    assert player.pipeline is pipeline
    assert player.voice == "af_heart"


def test_language_change_reuses_the_loaded_model(monkeypatch):
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(models, "KPipeline", fake_pipeline)
    model = object()
    player = make_tts_player()
    player.pipeline = types.SimpleNamespace(model=model)
    assert player.change_language("b", None)
    assert calls[0]["model"] is model
    assert player.pipeline.model is model
    assert player.language == "b"
    assert player.voice.startswith("b")
