from pathlib import Path

import pytest

from utils import SRTEntry, parse_srt_file, parse_srt_timestamp

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("timestamp", "seconds"),
    [
        ("00:00:00,000", 0.0),
        ("00:00:03,500", 3.5),
        ("00:01:02,250", 62.25),
        ("01:00:00,001", 3600.001),
    ],
)
def test_parse_srt_timestamp(timestamp, seconds):
    assert parse_srt_timestamp(timestamp) == pytest.approx(seconds)


def test_parse_example_file():
    entries = parse_srt_file(str(REPO_ROOT / "test_subtitle.srt"))

    assert len(entries) == 5
    assert entries[0] == SRTEntry(
        1, 0.0, 3.0, "Hello and welcome to our demonstration."
    )
    assert entries[-1].index == 5
    assert entries[-1].start_time == pytest.approx(17.5)
    assert entries[-1].end_time == pytest.approx(20.0)


def test_multiline_text_is_joined(tmp_path):
    srt = tmp_path / "multi.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nfirst line\nsecond line\n",
        encoding="utf-8",
    )

    [entry] = parse_srt_file(str(srt))

    assert entry.text == "first line\nsecond line"


def test_malformed_blocks_are_skipped(tmp_path):
    srt = tmp_path / "bad.srt"
    srt.write_text(
        "x\n00:00:01,000 --> 00:00:02,000\nbad index\n\n"
        "2\nnot a timestamp\nbad time\n\n"
        "3\n00:00:01,000\n\n"
        "4\n00:00:05,000 --> 00:00:06,000\nkept\n",
        encoding="utf-8",
    )

    entries = parse_srt_file(str(srt))

    assert [e.index for e in entries] == [4]
    assert entries[0].text == "kept"


def test_empty_file(tmp_path):
    srt = tmp_path / "empty.srt"
    srt.write_text("", encoding="utf-8")

    assert parse_srt_file(str(srt)) == []


def test_utf8_bom_keeps_first_entry(tmp_path):
    srt = tmp_path / "bom.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nfirst\n", encoding="utf-8-sig")

    entries = parse_srt_file(str(srt))

    assert [e.index for e in entries] == [1]


def write_srt(tmp_path, content):
    path = tmp_path / "sub.srt"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_parse_dot_milliseconds_and_long_hours(tmp_path):
    entries = parse_srt_file(
        write_srt(tmp_path, "1\n100:00:01.500 --> 100:00:02.000\nHi\n")
    )
    assert entries == [SRTEntry(1, 360001.5, 360002.0, "Hi")]


def test_skip_entry_ending_before_start(tmp_path):
    entries = parse_srt_file(
        write_srt(
            tmp_path,
            "1\n00:00:05,000 --> 00:00:01,000\nBad\n\n"
            "2\n00:00:06,000 --> 00:00:07,000\nGood\n",
        )
    )
    assert [entry.text for entry in entries] == ["Good"]
