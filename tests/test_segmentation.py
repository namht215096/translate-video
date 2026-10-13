import pytest

from translate_video.segmentation import Segmenter


def test_unstable_comma_is_not_committed():
    segmenter = Segmenter()
    assert segmenter.update("I think,") == []
    assert segmenter.update("I thought he would come.") == []
    assert segmenter.update("I thought he would come. Tomorrow") == ["I thought he would come."]
    assert segmenter.update("I thought he would come. Tomorrow", final=True) == ["Tomorrow"]


def test_no_punctuation_has_a_word_limit_and_final_flush():
    segmenter = Segmenter(max_words=3)
    assert segmenter.update("one two three four") == []
    assert segmenter.update("one two three four") == ["one two three"]
    assert segmenter.update("one two three four", final=True) == ["four"]
    assert segmenter.update("one two three four", final=True) == []


def test_committed_revision_fails_instead_of_duplicating_audio():
    segmenter = Segmenter()
    segmenter.update("Hello, world")
    assert segmenter.update("Hello, everyone") == ["Hello,"]
    with pytest.raises(RuntimeError, match="already committed"):
        segmenter.update("Goodbye, everyone", final=True)


def test_long_punctuated_sentence_is_bounded():
    segmenter = Segmenter(max_words=3)
    assert segmenter.update("one two three four five six seven.", final=True) == [
        "one two three", "four five six", "seven.",
    ]


def stable(segmenter, text, final=False):
    """Feed the same text twice so every word counts as stable."""
    segmenter.update(text)
    return segmenter.update(text, final=final)


def test_semantic_keeps_a_short_clause_with_its_sentence():
    text = "Yesterday, I walked to the market and bought some fresh fruit."
    assert stable(Segmenter(), text) == ["Yesterday,", "I walked to the market and bought some fresh fruit."]
    assert stable(Segmenter(mode="semantic"), text) == [text]


def test_semantic_cuts_a_comma_only_after_enough_words():
    text = "If you really want to learn faster than everybody else, you need to practice every single day."
    assert stable(Segmenter(mode="semantic"), text) == [
        "If you really want to learn faster than everybody else,",
        "you need to practice every single day.",
    ]


def test_semantic_cuts_a_long_unpunctuated_run_before_a_clause_opener():
    text = "we tested three different methods in the lab last week and the last one worked best for us"
    assert stable(Segmenter(mode="semantic"), text) == [
        "we tested three different methods in the lab last week",
    ]
    # still a bounded fallback when nothing looks like a clause opener
    words = " ".join(f"w{i}" for i in range(30))
    assert len(stable(Segmenter(mode="semantic"), words)[0].split()) == 24


def test_semantic_waits_for_stability_and_flushes_at_the_end():
    segmenter = Segmenter(mode="semantic")
    assert segmenter.update("I think we should start") == []
    assert segmenter.update("I think we should start now", final=True) == ["I think we should start now"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Segmenter(mode="vibes")
