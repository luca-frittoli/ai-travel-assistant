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
    obj = None
    # Try every "{" in turn: prose before the JSON may itself contain braces.
    start = raw.find("{")
    while start != -1:
        try:
            candidate, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(candidate, dict) and "verdict" in candidate:
                obj = candidate
                break
        start = raw.find("{", start + 1)

    if obj is None:
        raise ValueError("no JSON object with a verdict found in judge output")

    verdict = obj["verdict"]
    if not isinstance(verdict, str) or verdict.strip().lower() not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is invalid")

    return {"reasoning": obj.get("reasoning", ""), "verdict": verdict.strip().lower()}


def is_hit(question: dict, retrieved: list[tuple[float, dict]]) -> bool | None:
    """True if any retrieved chunk has the question's city and section.

    `retrieved` is the output of DenseRetriever.search: [(score, chunk), ...].
    Returns None for unanswerable questions (question["section"] is None),
    since there is no "right" chunk to find.
    """
    if question["section"] is None:
        return None
    for _, chunk in retrieved:
        if question["city"] == chunk["city"] and question["section"] in chunk["sections"]:
            return True
    return False


def hit_rate(questions: list[dict], retrieved_per_question: list[list[tuple[float, dict]]]) -> float:
    """Fraction of answerable questions with a hit (unanswerable ones are skipped).

    Raises ValueError if the two lists have different lengths, or if there is
    no answerable question.
    """
    if len(questions) != len(retrieved_per_question):
        raise ValueError(f"{len(questions)} questions and {len(retrieved_per_question)} retrieval results")

    answerable_questions = []
    retrieved_per_answerable_question = []
    for q, r in zip(questions, retrieved_per_question):
        if q["section"] is not None:
            answerable_questions.append(q)
            retrieved_per_answerable_question.append(r)

    if len(answerable_questions) == 0:
        raise ValueError(f"{len(answerable_questions)} answerable questions")

    hits = [q for q, r in zip(answerable_questions, retrieved_per_answerable_question) if is_hit(q, r)]

    return len(hits) / len(answerable_questions)


def judge_answers(caller, answers: list[dict]) -> list[dict]:
    """Ask the judge to grade every answer.

    `answers` items have "question", "expected_answer", "model_answer" (the
    format written by plain_llm_test / retrieved_chunks_test). `caller` is an
    LLMCaller (anything with a .judge(question, expected_answer, system_answer) method).

    Returns new dicts (inputs are not mutated) with all original keys plus
    "verdict" and "reasoning". If the judge output can't be parsed, the verdict
    is "error" and "reasoning" holds the raw judge output, so you can inspect it.
    """
    judgements = []
    for answer in answers:
        judgement = caller.judge(answer["question"], answer["expected_answer"], answer["model_answer"])
        try:
            judgement = parse_verdict(judgement)
        except ValueError:
            judgement = {
                "verdict": "error",
                "reasoning": judgement,
            }
        judgements.append({**answer, "verdict": judgement["verdict"], "reasoning": judgement["reasoning"]})
    return judgements


def summarize_verdicts(judged: list[dict]) -> dict:
    """Aggregate judged answers.

    Returns:
        {"n": total items,
         "correct": int, "partial": int, "incorrect": int, "error": int,
         "accuracy": correct / n_valid,
         "score": (correct + 0.5 * partial) / n_valid}
    where n_valid excludes "error" items. accuracy and score are None when n_valid == 0.
    """
    aggregated_judgements = {
        "n": 0,
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
        "error": 0,
    }
    for judgement in judged:
        aggregated_judgements["n"] += 1
        aggregated_judgements[judgement["verdict"]] += 1

    correct = aggregated_judgements["correct"]
    partial = aggregated_judgements["partial"]
    n_valid = aggregated_judgements["n"] - aggregated_judgements["error"]

    aggregated_judgements["accuracy"] = correct / n_valid if n_valid > 0 else None
    aggregated_judgements["score"] = (correct + 0.5 * partial) / n_valid if n_valid > 0 else None

    return aggregated_judgements
