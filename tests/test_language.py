from translate_video.language import clean_asr_text, contains_cjk, looks_vietnamese, nllb_source_code, prepare_for_translation


def test_vietnamese_is_detected():
    assert looks_vietnamese("Khi tôi bắt đầu nghe, cậu ấy thậm chí còn không lớn đến thế.")
    assert looks_vietnamese("chắc vô bấm search kiếm lên đây mày.")


def test_english_and_loanwords_are_not():
    assert not looks_vietnamese("He wasn't even that big when I started listening to him,")
    assert not looks_vietnamese("We met at the café near the résumé office.")
    assert not looks_vietnamese("Okay.")
    assert not looks_vietnamese("")


def test_source_code_follows_the_script_of_the_sentence():
    assert nllb_source_code("你可能无法相信，最近马斯克在硅谷接受了专访。") == "zho_Hans"
    assert nllb_source_code("He really cares about it.") == "eng_Latn"
    assert nllb_source_code("Girl,自来熟的样子显然是将这个名叫佐佐木的大叔吓得够呛") == "zho_Hans"
    assert nllb_source_code("今日はいい天気ですね。") == "jpn_Jpan"
    assert nllb_source_code("안녕하세요 여러분") == "kor_Hang"
    assert nllb_source_code("I bought an iPhone 15 在东京") == "eng_Latn"          # mostly English: keep the default
    assert nllb_source_code("123 !!!", default="vie_Latn") == "vie_Latn"


def test_contains_cjk():
    assert contains_cjk("成绩靠的就是中国团队──")
    assert contains_cjk("かな")
    assert not contains_cjk("Anh ấy thực sự quan tâm.")


def test_trailing_clause_marks_are_dropped_before_translation():
    assert prepare_for_translation("他认为中国拥有非常深厚的制造业基础，") == "他认为中国拥有非常深厚的制造业基础"
    assert prepare_for_translation("论文、") == "论文"
    assert prepare_for_translation("Yesterday, ") == "Yesterday"
    assert prepare_for_translation("He left. ") == "He left."          # a full stop is kept (trailing space goes)
    assert prepare_for_translation("美国加州。") == "美国加州。"
    assert prepare_for_translation("，") == "，"                       # never return an empty string


def test_qwen_prompt_leaks_are_removed_from_asr_text():
    assert clean_asr_text("�system user assistant language English<asr_text>Just,") == "Just,"
    assert clean_asr_text("� Musk in Silicon Valley.") == "Musk in Silicon Valley."
    assert clean_asr_text("language Chinese<asr_text>你好，世界") == "你好，世界"
    assert clean_asr_text("system user assistant language English Just,") == "Just,"
    assert clean_asr_text("  He really cares about.  ") == "He really cares about."
    assert clean_asr_text("") == ""


def test_ordinary_text_that_mentions_these_words_is_kept():
    assert clean_asr_text("The system user said hello.") == "The system user said hello."
    assert clean_asr_text("You are the assistant.") == "You are the assistant."
