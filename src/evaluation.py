"""Stage 6: evaluation helpers (LLM-as-judge + retrieval hit rate).

These are stubs: the contracts are described in the docstrings and pinned down
by tests/test_evaluation.py. Implement them until `pytest tests/test_evaluation.py` passes.
"""

import json

VERDICTS = ("correct", "partial", "incorrect")
UNANSWERABLE = "Not in the guide."


def parse_verdict(raw: str) -> dict:
    """Turn the judge's raw text into {"reasoning": str, "verdict": str}.

    - Accepts a bare JSON object, a JSON object inside ```json ... ``` fences,
      or a JSON object surrounded by extra prose.
    - "verdict" is normalised (strip + lowercase) and must be one of VERDICTS.
    - Missing "reasoning" becomes "".
    - Raises ValueError if no JSON object can be found or the verdict is invalid.
    """
    decoder = json.JSONDecoder()
    start = raw.index("{")  # first "{" in the text
    obj, end = decoder.raw_decode(raw, start)

    if obj.get("verdict", "").strip().lower() not in VERDICTS:
        raise ValueError(f"verdict {obj.get('verdict', '-')} is invalid")
    else:
        obj["verdict"] = obj["verdict"].strip().lower()
    if "reasoning" not in obj:
        obj["reasoning"] = ""

    return obj




def is_hit(question: dict, retrieved: list[tuple[float, dict]]) -> bool | None:
    """True if any retrieved chunk has the question's city and section.

    `retrieved` is the output of DenseRetriever.search: [(score, chunk), ...].
    Returns None for unanswerable questions (question["section"] is None),
    since there is no "right" chunk to find.
    """
    raise NotImplementedError


def hit_rate(questions: list[dict], retrieved_per_question: list[list[tuple[float, dict]]]) -> float:
    """Fraction of answerable questions with a hit (unanswerable ones are skipped).

    Raises ValueError if the two lists have different lengths, or if there is
    no answerable question.
    """
    raise NotImplementedError


def judge_answers(caller, answers: list[dict]) -> list[dict]:
    """Ask the judge to grade every answer.

    `answers` items have "question", "expected_answer", "model_answer" (the
    format written by plain_llm_test / retrieved_chunks_test). `caller` is an
    LLMCaller (anything with a .judge(question, expected_answer, system_answer) method).

    Returns new dicts (inputs are not mutated) with all original keys plus
    "verdict" and "reasoning". If the judge output can't be parsed, the verdict
    is "error" and "reasoning" holds the raw judge output, so you can inspect it.
    """
    raise NotImplementedError


def summarize_verdicts(judged: list[dict]) -> dict:
    """Aggregate judged answers.

    Returns:
        {"n": total items,
         "correct": int, "partial": int, "incorrect": int, "error": int,
         "accuracy": correct / n_valid,
         "score": (correct + 0.5 * partial) / n_valid}
    where n_valid excludes "error" items. accuracy and score are None when n_valid == 0.
    """
    raise NotImplementedError
