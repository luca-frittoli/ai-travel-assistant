"""Judge calibration: does the LLM judge agree with hand-labelled verdicts?

This is the automated half of "spot-check the judge's verdicts by hand".
Each case in tests/data/judge_calibration.json is an answer whose verdict a human
decided. Failures here are not necessarily bugs in your code: they show where the
judge prompt (JUDGE_RULES) or the judge model disagrees with you. Add a case
every time you catch the judge making a mistake while spot-checking.

Needs Ollama running: pytest -m integration tests/test_judge_calibration.py
"""
import json
from pathlib import Path

import pytest

from src.config import JUDGE_LLM
from src.evaluation import VERDICTS, parse_verdict
from src.LLMCaller import LLMCaller
from tests.conftest import ollama_available

CASES = json.loads((Path(__file__).parent / "data" / "judge_calibration.json").read_text(encoding="utf-8"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ollama_available(JUDGE_LLM), reason="Ollama server/model not available"),
]


def test_cases_are_well_formed():
    for case in CASES:
        assert case["verdict"] in VERDICTS, case


@pytest.fixture(scope="module")
def caller():
    return LLMCaller(JUDGE_LLM)


@pytest.mark.parametrize("case", CASES, ids=[c["note"] for c in CASES])
def test_judge_agrees_with_human(caller, case):
    raw = caller.judge(case["question"], case["expected_answer"], case["model_answer"])
    result = parse_verdict(raw)
    assert result["verdict"] == case["verdict"], f"judge said: {result['reasoning']}"
