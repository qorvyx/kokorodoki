import pytest

from utils import (
    MAX_SENTENCE_LEN_NO_SPACES,
    get_default_voice,
    merge_short_sentences,
    split_by_words,
    split_long_sentence,
    split_text_to_sentences,
)


def test_split_by_words_short_chunk_unchanged():
    assert split_by_words("hello world", 50) == ["hello world"]


def test_split_by_words_respects_max_len():
    text = " ".join(["word"] * 40)

    chunks = split_by_words(text, 30)

    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks) == text


def test_split_by_words_breaks_overlong_word():
    chunks = split_by_words("a " + "x" * 25, 10)

    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks).replace(" ", "") == "a" + "x" * 25


def test_split_long_sentence_short_unchanged():
    assert split_long_sentence("Short sentence.") == ["Short sentence."]


def test_split_long_sentence_uses_punctuation_and_limit():
    clause = "this clause has a reasonable number of words in it"
    sentence = ", ".join([clause] * 12) + "."

    chunks = split_long_sentence(sentence, max_len=120, min_len=20)

    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    assert "".join(chunks).replace(" ", "") == sentence.replace(" ", "")


def test_split_long_sentence_without_punctuation():
    sentence = " ".join(["word"] * 100)

    chunks = split_long_sentence(sentence, max_len=60)

    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks) == sentence


def test_merge_short_sentences():
    assert merge_short_sentences([]) == []
    assert merge_short_sentences(["a. ", "b. ", "c" * 60], min_len=5) == [
        "a. b. ",
        "c" * 60,
    ]


def test_split_long_sentence_uses_cjk_commas():
    sentence = "、".join(["あいうえおかきくけこ"] * 12)

    chunks = split_long_sentence(sentence, max_len=40, min_len=5)

    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks) == sentence


def test_split_text_to_sentences_non_latin():
    nltk = pytest.importorskip("nltk")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pytest.skip("nltk punkt_tab data not installed")

    text = "今日は晴れです。明日は雨です！" + "あ" * 150 + "。"

    sentences = split_text_to_sentences(text, "english")

    assert sentences[:2] == ["今日は晴れです。", "明日は雨です！"]
    assert all(len(s) <= MAX_SENTENCE_LEN_NO_SPACES for s in sentences)
    assert "".join(sentences) == text


@pytest.mark.parametrize(
    ("language", "voice"),
    [("a", "af_heart"), ("b", "bf_alice"), ("f", "ff_siwis"), ("j", "jf_alpha")],
)
def test_default_voice_follows_language(language, voice):
    assert get_default_voice(language, "af_heart") == voice
