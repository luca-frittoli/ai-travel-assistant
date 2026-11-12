"""Tests for stage 6 (src/evaluation.py). These describe the behaviour to implement."""
import copy

import pytest

from src.evaluation import (
    UNANSWERABLE,
    VERDICTS,
    hit_rate,
    is_hit,
    judge_agreement,
    judge_answers,
    parse_verdict,
    summarize_verdicts,
)


# --------------------------------------------------------------------------- parse_verdict

class TestParseVerdict:
    def test_plain_json(self):
        assert parse_verdict('{"reasoning": "matches", "verdict": "correct"}') == {
            "reasoning": "matches", "verdict": "correct"}

    def test_json_in_code_fence(self):
        raw = '```json\n{"reasoning": "half", "verdict": "partial"}\n```'
        assert parse_verdict(raw)["verdict"] == "partial"

    def test_json_surrounded_by_prose(self):
        raw = 'Here is my evaluation:\n{"reasoning": "wrong", "verdict": "incorrect"}\nHope it helps!'
        assert parse_verdict(raw)["verdict"] == "incorrect"

    def test_verdict_is_normalised(self):
        assert parse_verdict('{"reasoning": "", "verdict": "  Correct "}')["verdict"] == "correct"

    def test_braces_inside_reasoning(self):
        raw = '{"reasoning": "the answer {mostly} matches", "verdict": "partial"}'
        assert parse_verdict(raw) == {"reasoning": "the answer {mostly} matches", "verdict": "partial"}

    def test_missing_reasoning_defaults_to_empty(self):
        assert parse_verdict('{"verdict": "correct"}') == {"reasoning": "", "verdict": "correct"}

    @pytest.mark.parametrize("raw", [
        "",
        "The answer is correct.",
        '{"reasoning": "x", "verdict": "mostly right"}',
        '{"reasoning": "x"}',
        '{"reasoning": "x", "verdict": "correct"',  # truncated
    ])
    def test_invalid_output_raises(self, raw):
        with pytest.raises(ValueError):
            parse_verdict(raw)

    def test_verdicts_constant(self):
        assert set(VERDICTS) == {"correct", "partial", "incorrect"}


# --------------------------------------------------------------------------- is_hit / hit_rate

def chunk(city, *sections):
    return {"id": f"{city}-x", "city": city, "sections": list(sections), "text": ""}


Q_LISBON_GET_IN = {"question": "q", "expected_answer": "a", "city": "Lisbon", "section": "Get in"}
Q_UNANSWERABLE = {"question": "q", "expected_answer": UNANSWERABLE, "city": "Bilbao", "section": None}


class TestIsHit:
    def test_hit_when_city_and_section_match(self):
        retrieved = [(0.9, chunk("Lisbon", "Eat")), (0.8, chunk("Lisbon", "Get in"))]
        assert is_hit(Q_LISBON_GET_IN, retrieved) is True

    def test_section_from_other_city_is_not_a_hit(self):
        assert is_hit(Q_LISBON_GET_IN, [(0.9, chunk("Dublin", "Get in"))]) is False

    def test_right_city_wrong_section_is_not_a_hit(self):
        assert is_hit(Q_LISBON_GET_IN, [(0.9, chunk("Lisbon", "Eat"))]) is False

    def test_chunk_spanning_several_sections(self):
        # fixed-size chunks can span sections
        assert is_hit(Q_LISBON_GET_IN, [(0.9, chunk("Lisbon", "Understand", "Get in"))]) is True

    def test_empty_retrieval(self):
        assert is_hit(Q_LISBON_GET_IN, []) is False

    def test_unanswerable_returns_none(self):
        assert is_hit(Q_UNANSWERABLE, [(0.9, chunk("Bilbao", "Do"))]) is None


class TestHitRate:
    def test_skips_unanswerable(self):
        questions = [Q_LISBON_GET_IN, Q_UNANSWERABLE, Q_LISBON_GET_IN]
        retrieved = [
            [(0.9, chunk("Lisbon", "Get in"))],
            [(0.9, chunk("Bilbao", "Do"))],
            [(0.9, chunk("Lisbon", "Eat"))],
        ]
        assert hit_rate(questions, retrieved) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(hit_rate([Q_LISBON_GET_IN], [[(0.9, chunk("Lisbon", "Get in"))]]), float)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            hit_rate([Q_LISBON_GET_IN], [])

    def test_only_unanswerable_raises(self):
        with pytest.raises(ValueError):
            hit_rate([Q_UNANSWERABLE], [[]])


# --------------------------------------------------------------------------- judge_answers

class ScriptedJudge:
    """Fake LLMCaller whose .judge returns pre-set outputs in order."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def judge(self, question, expected_answer, system_answer):
        self.calls.append((question, expected_answer, system_answer))
        return self.outputs.pop(0)


ANSWERS = [
    {"question": "Q1", "expected_answer": "R1", "model_answer": "A1"},
    {"question": "Q2", "expected_answer": "R2", "model_answer": "A2"},
]


class TestJudgeAnswers:
    def test_calls_judge_with_each_triple(self):
        judge = ScriptedJudge(['{"reasoning": "", "verdict": "correct"}'] * 2)
        judge_answers(judge, ANSWERS)
        assert judge.calls == [("Q1", "R1", "A1"), ("Q2", "R2", "A2")]

    def test_adds_verdict_and_reasoning_keeping_original_keys(self):
        judge = ScriptedJudge([
            '{"reasoning": "good", "verdict": "correct"}',
            '{"reasoning": "bad", "verdict": "incorrect"}',
        ])
        out = judge_answers(judge, ANSWERS)
        assert out[0] == {**ANSWERS[0], "verdict": "correct", "reasoning": "good"}
        assert out[1] == {**ANSWERS[1], "verdict": "incorrect", "reasoning": "bad"}

    def test_does_not_mutate_input(self):
        before = copy.deepcopy(ANSWERS)
        judge_answers(ScriptedJudge(['{"verdict": "correct"}'] * 2), ANSWERS)
        assert ANSWERS == before

    def test_unparseable_output_is_retried_once(self):
        judge = ScriptedJudge([
            "I think it's fine", '{"reasoning": "ok", "verdict": "correct"}',
            '{"reasoning": "", "verdict": "partial"}',
        ])
        out = judge_answers(judge, ANSWERS)
        assert [o["verdict"] for o in out] == ["correct", "partial"]
        assert len(judge.calls) == 3

    def test_unparseable_twice_becomes_error(self):
        judge = ScriptedJudge(["I think it's fine", "still prose", '{"reasoning": "", "verdict": "partial"}'])
        out = judge_answers(judge, ANSWERS)
        assert out[0]["verdict"] == "error"
        assert out[0]["reasoning"] == "still prose"
        assert out[1]["verdict"] == "partial"

    def test_empty_list(self):
        assert judge_answers(ScriptedJudge([]), []) == []


# --------------------------------------------------------------------------- summarize_verdicts

def judged(*verdicts):
    return [{"verdict": v} for v in verdicts]


class TestSummarizeVerdicts:
    def test_counts_and_scores(self):
        s = summarize_verdicts(judged("correct", "correct", "partial", "incorrect"))
        assert s["n"] == 4
        assert (s["correct"], s["partial"], s["incorrect"], s["error"]) == (2, 1, 1, 0)
        assert s["accuracy"] == pytest.approx(2 / 4)
        assert s["score"] == pytest.approx((2 + 0.5) / 4)

    def test_errors_count_as_incorrect(self):
        # An unjudged answer must not flatter the run: it scores 0, like "incorrect".
        s = summarize_verdicts(judged("correct", "error", "incorrect"))
        assert s["n"] == 3
        assert s["error"] == 1
        assert s["accuracy"] == pytest.approx(1 / 3)
        assert s["score"] == pytest.approx(1 / 3)

    def test_only_errors_scores_zero(self):
        s = summarize_verdicts(judged("error"))
        assert s["accuracy"] == 0 and s["score"] == 0

    def test_empty(self):
        s = summarize_verdicts([])
        assert s["n"] == 0 and s["accuracy"] is None


# --------------------------------------------------------------------------- judge_agreement

def labelled(question, answer, verdict):
    return {"question": question, "model_answer": answer, "verdict": verdict}


class TestJudgeAgreement:
    def test_compares_verdicts_on_matching_answers(self):
        judged_items = [labelled("Q1", "A1", "correct"), labelled("Q2", "A2", "partial")]
        labels = [labelled("Q1", "A1", "correct"), labelled("Q2", "A2", "incorrect")]
        assert judge_agreement(judged_items, labels) == {"n": 2, "agree": 1, "rate": 0.5}

    def test_ignores_answers_without_a_label(self):
        # Regenerated answers have different text, so old hand labels don't apply.
        judged_items = [labelled("Q1", "A1", "correct"), labelled("Q1", "new text", "correct")]
        labels = [labelled("Q1", "A1", "correct")]
        assert judge_agreement(judged_items, labels) == {"n": 1, "agree": 1, "rate": 1.0}

    def test_no_overlap_gives_none_rate(self):
        assert judge_agreement([labelled("Q", "A", "correct")], []) == {"n": 0, "agree": 0, "rate": None}

    def test_error_verdict_disagrees(self):
        assert judge_agreement([labelled("Q", "A", "error")], [labelled("Q", "A", "incorrect")])["agree"] == 0
