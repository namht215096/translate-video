from translate_video.language import looks_vietnamese


def test_vietnamese_is_detected():
    assert looks_vietnamese("Khi tôi bắt đầu nghe, cậu ấy thậm chí còn không lớn đến thế.")
    assert looks_vietnamese("chắc vô bấm search kiếm lên đây mày.")


def test_english_and_loanwords_are_not():
    assert not looks_vietnamese("He wasn't even that big when I started listening to him,")
    assert not looks_vietnamese("We met at the café near the résumé office.")
    assert not looks_vietnamese("Okay.")
    assert not looks_vietnamese("")
