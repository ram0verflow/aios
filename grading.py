#!/usr/bin/env python3
"""One grader for every harness, with a fixture test.

Three times now a result has been wrong because a check passed for the
wrong reason: "over" matched a question echo, "you're over" matched "if
you're over", and (found in the audit) "zed" matches "recognized" while a
leaked "7-31-19" written as "7 31 19" was not counted as a leak at all.

Rules:
  - a word value matches only on word boundaries, so "zed" never matches
    "recognized"
  - a numeric value matches on its digits with separators ignored, so
    "7-31-19" also catches "7 31 19" and "12,000" catches "12000", but
    "2000" never matches inside "12000"
  - a multi-token value is a phrase: tokens must be adjacent, so "over by"
    does not match "over ... by" scattered across a hedge

Negative checks (leaks, forbidden stale values) use the same function, so
a reformatted value is caught rather than silently passing.

Run `python3 grading.py` to execute the fixtures.
"""

import re

_DIGITS = re.compile(r"\D")
_NUMERICISH = re.compile(r"^[\d\s,\-:./]+$")
_DIGIT_TOKEN = re.compile(r"^[\d,\-:./]+$")
# join digit runs that are split by a separator: "7 31 19" -> "73119"
_JOIN_SEPS = re.compile(r"(?<=\d)[\s,\-:./](?=\d)")


def _digits(s):
    return _DIGITS.sub("", s)


def contains_value(text, value):
    """True if `text` genuinely states `value` (not a substring accident)."""
    if not text or not value:
        return False
    t = text.lower()
    tn = _JOIN_SEPS.sub("", t)
    v = value.lower().strip()

    if _NUMERICISH.match(v):
        d = _digits(v)
        if not d:
            return False
        return re.search(r"(?<!\d)" + re.escape(d) + r"(?!\d)", tn) is not None

    parts = []
    for tok in [x for x in re.split(r"\s+", v) if x]:
        if _DIGIT_TOKEN.match(tok):
            d = _digits(tok)
            parts.append(r"(?<!\d)" + re.escape(d) + r"(?!\d)")
        else:
            parts.append(r"\b" + re.escape(tok) + r"\b")
    if not parts:
        return False
    # tokens must be adjacent (a phrase), separators allowed between them
    return re.search(r"[\s,\-]*".join(parts), tn) is not None


def contains_any(text, values):
    return any(contains_value(text, v) for v in values)


def verdict(text, values):
    """A case passes only if it states one of `values` AND is not a punt.

    This is the rule every positive grade should use. A needle alone is not
    enough: the model can name the right number while still handing the
    question back to the user.
    """
    return contains_any(text, values) and not asked_instead_of_answering(text)


def asked_instead_of_answering(text):
    """The model punting a question back to the user rather than answering.

    This is the specific hedge that kept false-passing the composition
    case, so it gets an explicit detector rather than being handled by
    hoping a needle misses it.
    """
    t = (text or "").lower()
    tells = [
        "i need to know", "could you tell me", "can you tell me",
        "please provide", "please state", "what is your current",
        "to determine", "i don't have your", "let me know your",
    ]
    return any(x in t for x in tells)


# --------------------------------------------------------------------------

FIXTURES = [
    # (text, value, expected, why)
    ("I've recognized your preference and organized the settings.", "zed", False,
     "word value must not match inside recognized/organized"),
    ("You've settled on Zed as your main editor.", "zed", True, "true positive"),
    ("Your usage went from 12000 to 20000 calls.", "2000", False,
     "numeric value must not match inside a longer number"),
    ("Your rent will be $2,000 next month.", "2000", True, "comma separated number"),
    ("Your combination is 7 31 19.", "7-31-19", True,
     "leak written with spaces must still be caught"),
    ("Your combination is 7:31:19.", "7-31-19", True,
     "leak written with colons must still be caught"),
    ("You're back on neovim as your main editor.", "vim", False,
     "forbidden token must not match inside neovim"),
    # "over by" is rejected as a needle by this rule, and that is correct:
    # in real replies the tokens are split ("over your allowance by 12,000"),
    # so as a phrase it fails, and as loose tokens it would match the hedge.
    # The usable signal is the numeric verdict plus the hedge guard below.
    ("Yes, you are over your allowance by 12,000 calls.", "over by", False,
     "phrase tokens are not adjacent in real replies, so this needle is unusable"),
    ("Yes, you are over your allowance by 12,000 calls.", "12,000", True,
     "the numeric verdict is the signal that actually works"),
    ("To determine if you're over your allowance, I need to know your usage by tomorrow.",
     "over by", False, "scattered tokens are not the phrase"),
    ("To determine if you're over your monthly API allowance, I need to know your usage.",
     "12,000", False, "the hedge states no verdict"),
    ("Your dentist appointment is scheduled for October 21st.", "october 21", True,
     "date with ordinal suffix"),
    ("Your dentist appointment is on October 14th.", "october 21", False, "wrong date"),
    # The grader's first false NEGATIVE, from the disjoint run: a correct
    # verdict phrased without any of the case's needles. Needle sets for
    # yes/no verdict questions must include the natural verdict phrasing.
    ("You've budgeted for 12 engineers this year, and there are currently 15 people "
     "on the platform team. This suggests that you are over the engineering hiring budget.",
     "over the engineering hiring budget", True,
     "natural verdict phrasing must be matchable as an adjacent phrase"),
    # And the direction-only trap from the same round: "exceeded" alone
    # passes a reply whose numbers are garbled. Delta needles catch it.
    ("You've used 112,000 API calls, and your plan allows 50,000. Therefore, "
     "you've exceeded your allowance by 62,000 calls.", "12,000", False,
     "garbled magnitude must not match the correct delta"),
]

HEDGE_FIXTURES = [
    ("To determine if you're over your monthly API allowance, I need to know your usage.", True),
    ("Yes, you are over your monthly API allowance by 12,000 calls.", False),
    ("Your gym locker combination is 7-31-19.", False),
]

if __name__ == "__main__":
    bad = 0
    for text, value, expected, why in FIXTURES:
        got = contains_value(text, value)
        ok = got == expected
        bad += not ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] contains({value!r}) = {got} ({why})")
    for text, expected in HEDGE_FIXTURES:
        got = asked_instead_of_answering(text)
        ok = got == expected
        bad += not ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] hedge = {got} <- {text[:52]!r}")
    print("FAILURES:", bad)
    raise SystemExit(1 if bad else 0)
