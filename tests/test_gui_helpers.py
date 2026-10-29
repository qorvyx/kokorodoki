from utils import (
    format_speed,
    get_gui_themes,
    has_text,
    is_image_file,
    pick_font,
    reading_padding,
    sentence_progress,
)


def test_old_theme_numbers_map_to_dark_and_light():
    themes = get_gui_themes()

    assert [themes[n] for n in range(1, 5)] == ["dark"] * 4
    assert [themes[n] for n in range(5, 8)] == ["light"] * 3
    assert 0 not in themes and 8 not in themes


def test_sentence_progress_without_sentences():
    assert sentence_progress(None, 0) == ("", 0.0)
    assert sentence_progress(3, 0) == ("", 0.0)


def test_sentence_progress_before_playback():
    assert sentence_progress(None, 12) == ("12 sentences", 0.0)
    assert sentence_progress(None, 1) == ("1 sentence", 0.0)


def test_sentence_progress_counts_from_one():
    assert sentence_progress(0, 4) == ("Sentence 1 of 4", 0.25)
    assert sentence_progress(3, 4) == ("Sentence 4 of 4", 1.0)


def test_sentence_progress_clamps_out_of_range():
    assert sentence_progress(9, 4) == ("Sentence 4 of 4", 1.0)
    assert sentence_progress(-2, 4) == ("Sentence 1 of 4", 0.25)


def test_pick_font_takes_first_installed():
    available = ["DejaVu Sans", "Cantarell", "Inter"]

    assert pick_font(available, ["Inter", "Segoe UI", "Cantarell"]) == "Inter"
    assert pick_font(available, ["Segoe UI", "Cantarell"]) == "Cantarell"


def test_pick_font_ignores_case_and_falls_back_to_last():
    assert pick_font(["noto serif"], ["Noto Serif", "Georgia"]) == "noto serif"
    assert pick_font([], ["Georgia", "DejaVu Serif"]) == "DejaVu Serif"


def test_has_text_ignores_whitespace():
    assert not has_text("")
    assert not has_text(" \n\t ")
    assert has_text(" Hello ")


def test_is_image_file():
    assert is_image_file("/tmp/scan.PNG")
    assert is_image_file("page.tiff")
    assert not is_image_file("notes.txt")
    assert not is_image_file("png")


def test_format_speed():
    assert format_speed(1.2) == "1.2×"
    assert format_speed(1.0) == "1×"
    assert format_speed(1.2000000001) == "1.2×"
    assert format_speed(0.75) == "0.75×"


def test_reading_padding_centres_the_measure():
    assert reading_padding(1000, 600, 28) == 200
    assert reading_padding(500, 600, 28) == 28
