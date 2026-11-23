"""10 sample English questions plus Hindi checks with expected facts.

Each case lists substrings that must appear in the answer. The test asserts both
the individual cases and the overall 8/10 bar from the project brief.
"""

from __future__ import annotations

import pytest

SAMPLES = [
    {
        "q": "How do I cancel my e-ticket and when will I get the refund?",
        "expect": ["booked ticket history", "refund"],
        "lang": "en",
    },
    {
        "q": "What are the cancellation charges for AC 3 Tier more than 48 hours before departure?",
        "expect": ["180", "3 tier"],
        "lang": "en",
    },
    {
        "q": "What are the Tatkal booking timings and charges for sleeper?",
        "expect": ["10:00", "11:00", "100"],
        "lang": "en",
    },
    {
        "q": "Can I book a Tatkal ticket in First AC?",
        "expect": ["except first ac"],
        "lang": "en",
    },
    {
        "q": "Which ID proof can I carry for an e-ticket?",
        "expect": ["passport", "pan card"],
        "lang": "en",
    },
    {
        "q": "What happens if I forget to carry my ID card during the journey?",
        "expect": ["without a ticket"],
        "lang": "en",
    },
    {
        "q": "How can I check my PNR status?",
        "expect": ["pnr status", "139"],
        "lang": "en",
    },
    {
        "q": "What is the difference between SL and 3A?",
        "expect": ["sleeper class", "ac 3 tier"],
        "lang": "en",
    },
    {
        "q": "What is the ladies quota?",
        "expect": ["ladies quota"],
        "lang": "en",
    },
    {
        "q": "How do I file TDR if the train is cancelled?",
        "expect": ["automatic full refund", "72 hours"],
        "lang": "en",
    },
]

HINDI = [
    {"q": "टिकट रद्द करने पर रिफंड कब मिलता है?", "expect": ["रिफंड"]},
    {"q": "तत्काल बुकिंग का समय क्या है?", "expect": ["10:00", "11:00"]},
    {"q": "ई-टिकट के साथ कौन सा आईडी प्रूफ चाहिए?", "expect": ["पासपोर्ट", "ड्राइविंग"]},
    {"q": "3A और SL में क्या अंतर है?", "expect": ["3a", "sl"]},
    {"q": "PNR स्टेटस कैसे चेक करें?", "expect": ["pnr status", "139"]},
]


def _evaluate(engine, case):
    result = engine.answer(case["q"], lang=case.get("lang", "auto"), force_offline=True)
    answer = result["answer"].lower()
    missing = [item for item in case["expect"] if item.lower() not in answer]
    return result, missing


@pytest.mark.parametrize("case", SAMPLES, ids=[c["q"][:38] for c in SAMPLES])
def test_sample_question(engine, case):
    result, missing = _evaluate(engine, case)
    assert not missing, f"Missing {missing} in answer:\n{result['answer']}"
    assert result["language"] == case["lang"]
    assert result["sources"], "every answered question should cite a source"


def test_at_least_8_of_10_correct(engine):
    correct = 0
    details = []
    for case in SAMPLES:
        result, missing = _evaluate(engine, case)
        ok = not missing
        correct += int(ok)
        details.append(f"{'PASS' if ok else 'FAIL'} {case['q'][:50]} missing={missing}")
    assert correct >= 8, "\n".join(details)


@pytest.mark.parametrize("case", HINDI, ids=[c["q"][:30] for c in HINDI])
def test_hindi_question(engine, case):
    result, missing = _evaluate(engine, case)
    assert result["language"] == "hi"
    assert not missing, f"Missing {missing} in Hindi answer:\n{result['answer']}"


def test_answers_are_short_point_form(engine):
    result = engine.answer("What are the cancellation charges for 2A?", force_offline=True)
    lines = [line for line in result["answer"].splitlines() if line.startswith("- ")]
    assert 1 <= len(lines) <= 10
    assert all(len(line) <= 300 for line in lines)
