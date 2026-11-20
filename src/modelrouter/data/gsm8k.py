"""Exact-match grading for GSM8K responses.

RouterBench's `grade-school-math` scores are a five-level grade (0, .25, .5, .75, 1) rather
than correctness: GPT-4's correct "72 clips" answer to the Natalia question scores 0.75, and only
5 of its 7,450 rows score 1.0.  We re-grade every response against the official GSM8K answers
(openai/gsm8k, MIT) so the quality label means "the final number is right".

Answer extraction.  The usual "last number in the response" rule (lm-eval's flexible extractor)
breaks on chat models, which end with a sentence that restates the givens: "Barry would need to
buy 6 bottles ... to last for 180 days" extracts 180.  The rule used here:

Find the concluding sentence (the last sentence or line containing a number), then take, in
order of preference:

1. an explicit cue in it: "the answer is 18", "Answer: 18", "#### 18", "\\boxed{18}";
2. its last equation result: "200 + 150 + 60 = 410 animals", "$22 / 2, which equals $11";
3. its last number that the response computed earlier ("672 / 4 = 168 ... each gets 168");
4. its first number that is not a given in the question ("buy 6 bottles ... for 180 days");
5. its first number.

Cues earlier in the text are ignored: models sometimes state an answer, then revise it.

`tests/test_gsm8k.py` covers the cases this was built from.
"""

from __future__ import annotations

import re
from fractions import Fraction

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_ANSWER_CUE = re.compile(
    rf"(?:answer(?: is)?|####|\\boxed\{{)\s*[:=]?\s*\**\s*\$?\s*({_NUMBER})", re.IGNORECASE
)
_ANY_NUMBER = re.compile(_NUMBER)
_EQUALS_RESULT = re.compile(
    rf"(?:=|\bequals\b|\bis equal to\b|\bwhich is\b)\s*\$?\s*({_NUMBER})", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_WS = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Whitespace- and case-insensitive key used to join RouterBench rows to GSM8K gold."""
    return _WS.sub(" ", text).strip().lower()


def _to_number(token: str) -> Fraction | None:
    token = token.replace(",", "").replace("$", "").strip().rstrip(".")
    if not token or token in {"-", "."}:
        return None
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def _numbers(text: str) -> list[Fraction]:
    out = []
    for tok in _ANY_NUMBER.findall(text):
        value = _to_number(tok)
        if value is not None:
            out.append(value)
    return out


_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
]  # fmt: skip
_TENS = {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
_OTHER = {"ninety": 90, "hundred": 100, "thousand": 1000, "dozen": 12, "twice": 2, "half": 2}
_NUMBER_WORDS = {w: i for i, w in enumerate(_UNITS)} | _TENS | _OTHER
_WORD = re.compile(r"[a-z]+")


def _givens(question: str) -> set[Fraction]:
    """Numbers stated in the question, including small ones written as words ("four")."""
    found = set(_numbers(question))
    found.update(
        Fraction(_NUMBER_WORDS[w]) for w in _WORD.findall(question.lower()) if w in _NUMBER_WORDS
    )
    return found


def gold_answer(solution: str) -> Fraction:
    """Parse the number after '####' in a GSM8K reference solution."""
    _, _, tail = solution.rpartition("####")
    value = _to_number(tail)
    if value is None:
        raise ValueError(f"no final answer in GSM8K solution: {solution[-80:]!r}")
    return value


def _parsed(tokens: list[str]) -> list[Fraction]:
    return [v for v in (_to_number(t) for t in tokens) if v is not None]


def extract_answer(response: str, question: str = "") -> Fraction | None:
    """The model's final numeric answer, or None if the response has no number."""
    sentences = [s for s in _SENTENCE_END.split(response) if _ANY_NUMBER.search(s)]
    if not sentences:
        return None
    final = sentences[-1]
    cued = _parsed(_ANSWER_CUE.findall(final))
    if cued:
        return cued[-1]
    results = _parsed(_EQUALS_RESULT.findall(final))
    if results:
        return results[-1]
    nums = _numbers(final)
    # A number the response computed earlier ("672 / 4 = 168" ... "each would get 168").
    computed = set(_parsed(_EQUALS_RESULT.findall(response)))
    restated = [v for v in nums if v in computed]
    if restated:
        return restated[-1]
    givens = _givens(question)
    fresh = [v for v in nums if v not in givens]
    if fresh:
        return fresh[0]
    return nums[0]


def extract_last_number(response: str) -> Fraction | None:
    """lm-eval-harness style 'flexible extract': the last number in the text (for comparison)."""
    nums = _numbers(response)
    return nums[-1] if nums else None


def is_correct(response: str, gold: Fraction, question: str = "") -> bool:
    predicted = extract_answer(response, question)
    return predicted is not None and predicted == gold
