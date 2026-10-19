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


def test_provisional_sentence_ending_can_change_before_it_is_spoken():
    segmenter = Segmenter(max_words=8, holdback_words=3)
    prefix = "one two three four five six seven eight"
    draft = prefix + " listening to him."
    assert segmenter.update(draft) == []
    assert segmenter.update(draft) == [prefix]
    assert segmenter.update(draft) == []
    corrected = prefix + " listening to him, but his solo"
    assert segmenter.update(corrected) == []
    assert segmenter.update(corrected) == ["listening to him,"]
    assert segmenter.update(corrected, final=True) == ["but his solo"]
