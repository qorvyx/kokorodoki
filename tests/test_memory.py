import sys
import types

import utils
from utils import (
    OcrReader,
    get_easyocr_languages,
    ocr_needs_load,
    over_memory_limit,
    parse_rss_mb,
    release_memory,
    restart_argv,
)

STATUS = """Name:\tpython
VmPeak:\t 4000000 kB
VmSize:\t 3900000 kB
VmHWM:\t 2100000 kB
VmRSS:\t 1536000 kB
RssAnon:\t 1400000 kB
"""


def test_parse_rss_mb_reads_vmrss():
    assert parse_rss_mb(STATUS) == 1500


def test_parse_rss_mb_missing_or_malformed():
    assert parse_rss_mb("Name:\tpython\nVmSize:\t 10 kB\n") is None
    assert parse_rss_mb("VmRSS:\n") is None
    assert parse_rss_mb("") is None


def test_over_memory_limit():
    assert over_memory_limit(3001, 3000)
    assert not over_memory_limit(3000, 3000)
    assert not over_memory_limit(2000, 3000)


def test_over_memory_limit_disabled_or_unknown():
    assert not over_memory_limit(99999, 0)
    assert not over_memory_limit(None, 3000)


def test_release_memory_runs_on_any_platform():
    release_memory()
    release_memory()


def test_get_easyocr_languages():
    assert get_easyocr_languages("a") == ["en"]
    assert get_easyocr_languages("z") == ["ch_sim"]
    assert get_easyocr_languages("x") == []


def test_ocr_needs_load():
    assert ocr_needs_load(None, ["en"])
    assert ocr_needs_load(["en"], ["fr"])
    assert not ocr_needs_load(["en"], ["en"])


def fake_easyocr(monkeypatch):
    """Install a fake easyocr module that records the readers it builds"""
    built = []

    class Reader:
        def __init__(self, languages):
            built.append(languages)
            self.languages = languages

        def readtext(self, image):
            return [(None, f"{self.languages[0]}:{image}", 1.0)]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=Reader))
    return built


def test_ocr_reader_loads_on_first_use(monkeypatch):
    built = fake_easyocr(monkeypatch)
    reader = OcrReader("a")
    assert built == []
    assert reader.readtext("img") == [(None, "en:img", 1.0)]
    reader.readtext("img")
    assert built == [["en"]]


def test_ocr_reader_rebuilds_only_when_ocr_language_changes(monkeypatch):
    built = fake_easyocr(monkeypatch)
    reader = OcrReader("a")
    reader.readtext("img")
    # American and British English share the OCR model
    reader.set_language("b")
    reader.readtext("img")
    assert built == [["en"]]
    reader.set_language("f")
    assert built == [["en"]]
    reader.readtext("img")
    assert built == [["en"], ["fr"]]


def test_ocr_reader_retries_after_failed_load(monkeypatch):
    built = fake_easyocr(monkeypatch)
    reader = OcrReader("a")
    reader.readtext("img")

    def broken(languages):
        raise RuntimeError("no model")

    monkeypatch.setattr(sys.modules["easyocr"], "Reader", broken)
    reader.set_language("f")
    try:
        reader.readtext("img")
    except RuntimeError:
        pass
    assert reader._reader is None
    assert utils.ocr_needs_load(reader._loaded, reader.languages)
    assert built == [["en"]]


def test_restart_argv_keeps_other_arguments():
    argv = ["main.py", "--daemon", "--port", "5611"]
    assert restart_argv(argv, "b", "bf_emma", 1.5) == [
        "main.py",
        "--daemon",
        "--port",
        "5611",
        "--language",
        "b",
        "--voice",
        "bf_emma",
        "--speed",
        "1.5",
    ]


def test_restart_argv_replaces_previous_settings():
    argv = ["kokorodoki", "-l", "a", "--daemon", "--voice=af_heart", "-s", "1.2"]
    once = restart_argv(argv, "b", "bf_emma", 1.0)
    assert once == [
        "kokorodoki",
        "--daemon",
        "--language",
        "b",
        "--voice",
        "bf_emma",
        "--speed",
        "1.0",
    ]
    # Restarting again doesn't grow the arguments
    assert restart_argv(once, "b", "bf_emma", 1.0) == once
