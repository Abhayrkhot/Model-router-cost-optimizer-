from fractions import Fraction

import pytest

from modelrouter.data import gsm8k


def test_gold_answer_parses_hash_marker():
    assert gsm8k.gold_answer("48/2 = <<48/2=24>>24\n48+24 = 72\n#### 72") == 72
    assert gsm8k.gold_answer("... #### 1,250") == 1250
    with pytest.raises(ValueError, match="no final answer"):
        gsm8k.gold_answer("no marker")


@pytest.mark.parametrize(
    ("response", "question", "expected"),
    [
        # Concluding sentence restates givens after the answer: first *new* number wins.
        (
            "360 capsules / 60 = 6 bottles\n\nBarry would need to buy 6 bottles to last for "
            "180 days.",
            "Barry needs vitamin D3 for 180 days. Each bottle has 60 capsules...",
            6,
        ),
        (
            "Therefore, Emma buys 30 containers of milk in 3 weeks.",
            "Emma buys 2 containers of milk every school day for 3 weeks.",
            30,
        ),
        # Final line is an equation: its last result wins over its operands.
        ("Total animals = 200 + 150 + 60 = 410 animals.", "...", 410),
        ("The price of one chair is $22 divided by 2, which equals $11.", "...", 11),
        # A value computed earlier and restated after an unrelated count.
        (
            "672 / 4 = 168\n\nIf they divided equally among the 4 of them, each would get 168.",
            "Four friends share strawberries...",
            168,
        ),
        # Explicit cue in the concluding sentence.
        ("Lots of work... The answer is $1,940.", "...", 1940),
        # A cue earlier in the text does not override a revised conclusion.
        ("The answer is 12. Wait, recheck: 7 * 3 = 21.\n\nSo she has 21 apples.", "...", 21),
        ("So the width is 7.5 feet.", "...", Fraction(15, 2)),
        ("I cannot determine this.", "...", None),
    ],
)
def test_extract_answer(response, question, expected):
    assert gsm8k.extract_answer(response, question) == expected


def test_last_number_extractor_fails_where_ours_succeeds():
    response = "Barry would need to buy 6 bottles to last for 180 days."
    assert gsm8k.extract_last_number(response) == 180
    assert gsm8k.is_correct(response, Fraction(6), "for 180 days")


def test_number_words_in_question_count_as_givens():
    # "four" is a given, so 4 is skipped and the computed 25 is taken.
    question = "There are four boxes with 100 pencils total. How many per box?"
    assert gsm8k.extract_answer("Each of the 4 boxes has 25 pencils.", question) == 25


def test_normalize_question_ignores_case_and_whitespace():
    assert gsm8k.normalize_question("  How many\n apples? ") == "how many apples?"
