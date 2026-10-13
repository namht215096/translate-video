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


MUSK = ("你可能无法相信，最近马斯克在硅谷接受了央视财经的一次长时间专访，而这次采访真正有意思的地方，"
        "并不是他说了多少漂亮话，而是他几乎把自己对未来十几年甚至二十年的。")


def test_chinese_text_without_spaces_is_cut_at_each_chinese_comma():
    assert stable(Segmenter(), MUSK) == [
        "你可能无法相信，",
        "最近马斯克在硅谷接受了央视财经的一次长时间专访，",
        "而这次采访真正有意思的地方，",
        "并不是他说了多少漂亮话，",
        "而是他几乎把自己对未来十几年甚至二十年的。",
    ]


def test_chinese_semantic_mode_keeps_clauses_together():
    assert stable(Segmenter(mode="semantic"), MUSK) == [
        "你可能无法相信，最近马斯克在硅谷接受了央视财经的一次长时间专访，",
        "而这次采访真正有意思的地方，并不是他说了多少漂亮话，",
        "而是他几乎把自己对未来十几年甚至二十年的。",
    ]


def test_chinese_waits_for_stability_before_committing():
    segmenter = Segmenter()
    assert segmenter.update("你可能无法相信，最近") == []
    assert segmenter.update("你可能无法相信，最近马斯克") == ["你可能无法相信，"]
    assert segmenter.update("你可能无法相信，最近马斯克", final=True) == ["最近马斯克"]


def test_chinese_length_limit_counts_a_character_as_half_a_word():
    text = "".join("字" * 10 + "" for _ in range(8))          # 80 characters, no punctuation
    pieces = stable(Segmenter(max_words=24), text)
    assert pieces and all(len(piece) == 48 for piece in pieces)       # 24 words == 48 characters
    assert stable(Segmenter(max_words=24), "字" * 47) == []


def test_other_chinese_and_closing_punctuation_end_a_sentence():
    assert stable(Segmenter(mode="semantic"), "他说：“我们走吧！”然后就离开了。", final=True) == [
        "他说：“我们走吧！”", "然后就离开了。"]
    assert stable(Segmenter(mode="semantic"), "真的吗？不可能！", final=True) == ["真的吗？", "不可能！"]


def test_chinese_clause_opener_splits_a_long_unpunctuated_run():
    text = "我们在实验室里测试了三种不同的方法而最后一种效果最好大家都很满意"
    pieces = stable(Segmenter(mode="semantic"), text)
    assert pieces == ["我们在实验室里测试了三种不同的方法"]


def test_mixed_english_and_chinese_keeps_original_spacing():
    text = "他用iPhone 15拍了这个视频，I think it looks great，真的。"
    assert stable(Segmenter(), text, final=True) == ["他用iPhone 15拍了这个视频，", "I think it looks great，", "真的。"]


def test_chinese_revision_of_committed_text_is_still_detected():
    segmenter = Segmenter()
    segmenter.update("你们好朋友们，世界")
    assert segmenter.update("你们好朋友们，世界") == ["你们好朋友们，"]
    with pytest.raises(RuntimeError, match="already committed"):
        segmenter.update("您们好朋友们，世界", final=True)


def test_chinese_enumeration_comma_splits_a_sentence():
    text = "首先想到的可能是论文、算法和芯片，"
    assert stable(Segmenter(), text) == ["首先想到的可能是论文、"]               # the 5-character tail waits...
    assert stable(Segmenter(), text, final=True) == ["首先想到的可能是论文、", "算法和芯片，"]   # ...and is flushed at the end
    # semantic mode keeps the list together: a 、 or comma only counts once the clause is long enough
    assert stable(Segmenter(mode="semantic"), text) == []
    assert stable(Segmenter(mode="semantic"), text + "但是马斯克一直非常重视制造能力。") == [
        "首先想到的可能是论文、算法和芯片，但是马斯克一直非常重视制造能力。"]


def test_very_short_chinese_pieces_are_merged_with_the_next_one():
    text = "AI人形机器人，Cybercab，中国芯片，电力，星舰，火星，脑机接口，甚至未来还有没有必要使用金钱？"
    assert stable(Segmenter(), text) == [
        "AI人形机器人，", "Cybercab，", "中国芯片，电力，", "星舰，火星，脑机接口，", "甚至未来还有没有必要使用金钱？"]


def test_short_chinese_piece_is_still_flushed_at_the_end_of_a_window():
    assert stable(Segmenter(), "好的。", final=True) == ["好的。"]
    assert stable(Segmenter(), "是的，", final=False) == []


def test_short_english_pieces_are_not_merged():
    assert stable(Segmenter(), "Yes, okay. Fine.", final=True) == ["Yes,", "okay.", "Fine."]
