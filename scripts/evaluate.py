"""Stage 6: end-to-end evaluation -> retrieval hit rate + answer correctness.

Run (needs Ollama):
    python -m scripts.evaluate                       # all systems on the merged eval set
    python -m scripts.evaluate --limit 5             # quick run: first 5 questions of each file
    python -m scripts.evaluate --judge-only out/answers_*.json   # re-judge saved answers
    python -m scripts.evaluate --judge-model qwen2.5:14b        # try another judge

Design notes:
- Retrieve once per question and reuse the result: hit rate is measured on the
  exact chunks the LLM sees (same K).
- Generation and judging are separate steps: answers are saved first, so you can
  re-judge them (--judge-only) after tweaking JUDGE_RULES without regenerating.
- The question files are merged into one eval set ("eval", 48 questions): one
  number per system, less noise than two small sets.
- Verdict files carry the judge's name (judged_<system>_eval__<judge>.json), so
  runs with different judges don't overwrite each other.
- Every run appends one line to out/results.jsonl (numbers + config): these are
  the two numbers every Stage 7 change should move.
- Unanswerable questions are reported separately: refusing well and answering
  well are different skills.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.DenseRetriever import DenseRetriever
from src.LLMCaller import LLMCaller
from src.config import QUESTION_RULES, DEFAULT_LLM, DEFAULT_EMBEDDING, JUDGE_LLM
from src.evaluation import hit_rate, judge_agreement, judge_answers, summarize_verdicts

DATA_FILES = ["data/sample_questions.json", "data/sample_questions_claude.json"]
EVAL_SET = "eval"                    # name of the merged set in file names and results
OUT = Path("out")
INDEXES = Path("indexes")
RESULTS_LOG = OUT / "results.jsonl"  # one line per run -> your tracked numbers
HAND_LABELS = Path("data/hand_labels.json")  # hand-graded verdicts -> judge agreement
K = 3                                # the same k the LLM actually sees


# --------------------------------------------------------------------------- systems
# Every system has the same shape: question dict -> (model_answer, retrieved or None)

def answer_plain(caller: LLMCaller, question: dict) -> tuple[str, None]:
    # Stage 1 baseline, no retrieval (see scripts/plain_llm_test.py)
    return caller.ask_llm(question['question'], system="keep the answer under 50 words"), None


def answer_rag(caller: LLMCaller, retriever: DenseRetriever, question: dict, use_city: bool=False) -> tuple[str, list]:
    city = None
    if use_city:
        city = question['city']
    retrieved = retriever.search(query=question['question'], k=K, city=city)

    question_rules = ""
    for _, chunk in retrieved:
        question_rules += f"{chunk['id']}: {chunk['text']}\n"

    return caller.ask_llm(question['question'], system=QUESTION_RULES + question_rules), retrieved


# --------------------------------------------------------------------------- helpers

def load_eval_set(files: list[str], limit: int | None = None) -> list[dict]:
    """Merge the question files, in order. `limit` keeps the first N of *each* file."""
    questions = []
    for data_file in files:
        questions += json.loads(Path(data_file).read_text(encoding="utf-8"))[:limit]
    return questions


def judged_path(run: str, judge_model: str, out: Path = OUT) -> Path:
    """out/judged_<run>__<judge>.json, with the model tag made filename-safe (qwen2.5:14b -> qwen2.5-14b)."""
    judge_tag = judge_model.replace(":", "-").replace("/", "-")
    return out / f"judged_{run}__{judge_tag}.json"


# --------------------------------------------------------------------------- pipeline steps

def generate(system_fn, questions: list[dict]) -> tuple[list[dict], list | None]:
    """Run a system over the questions.

    Returns (answers, retrieved): answers keep city/section so results can be
    sliced later; retrieved is None if the system doesn't retrieve.
    """
    answers = []
    retrieved = []
    n = len(questions)
    for i, question in enumerate(questions, start=1):
        print(f"  answering {i}/{n}", end="\r", flush=True)
        model_answer, retrieved_per_question = system_fn(question)
        answers.append({**question, "model_answer": model_answer})
        retrieved.append(retrieved_per_question)
    print()
    if all(r is None for r in retrieved):
        retrieved = None
    return answers, retrieved


def summarize_split(judged: list[dict]) -> dict:
    """Overall summary, plus answerable/unanswerable ones when items carry a section."""
    summary = summarize_verdicts(judged)
    if judged and all("section" in item for item in judged):
        summary["answerable"] = summarize_verdicts([j for j in judged if j["section"] is not None])
        summary["unanswerable"] = summarize_verdicts([j for j in judged if j["section"] is None])
    return summary


def load_hand_labels() -> list[dict]:
    return json.loads(HAND_LABELS.read_text(encoding="utf-8")) if HAND_LABELS.exists() else []


def evaluate_run(name: str, judge: LLMCaller,
                 questions: list[dict], answers: list[dict], retrieved: list | None) -> dict:
    hr = hit_rate(questions, retrieved) if retrieved is not None else None

    print("  judging...")
    judged = judge_answers(judge, answers)
    save_json(judged, judged_path(f"{name}_{EVAL_SET}", judge.model_name))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "dataset": EVAL_SET,
        "data_files": DATA_FILES,
        "n_questions": len(questions),
        "k": K if retrieved is not None else None,
        "llm": DEFAULT_LLM,
        "judge": judge.model_name,
        "embedding": DEFAULT_EMBEDDING if retrieved is not None else None,
        "hit_rate": hr,
        "judge_agreement": judge_agreement(judged, load_hand_labels()),
        **summarize_split(judged),
    }


def judge_only(path: Path, judge: LLMCaller) -> dict:
    """Re-judge an existing answers file (e.g. out/out.json) without regenerating."""
    answers = json.loads(path.read_text(encoding="utf-8"))
    print(f"judging {path} ({len(answers)} answers)...")
    judged = judge_answers(judge, answers)
    run = path.stem.removeprefix("answers_")
    save_json(judged, judged_path(run, judge.model_name))
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": f"judge_only:{path.name}",
        "dataset": run,
        "data_files": [str(path)],
        "n_questions": len(answers),
        "llm": DEFAULT_LLM,
        "judge": judge.model_name,
        "hit_rate": None,  # retrieval isn't re-run
        "judge_agreement": judge_agreement(judged, load_hand_labels()),
        **summarize_split(judged),
    }


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def log_result(record: dict) -> None:
    with RESULTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    line = (f"{record['name']:<12} {record['dataset']:<28} "
            f"hit_rate={fmt(record['hit_rate'])}  accuracy={fmt(record['accuracy'])}  "
            f"score={fmt(record['score'])}  errors={record['error']}/{record['n']}")
    agreement = record.get("judge_agreement") or {}
    if agreement.get("n"):
        line += f"  judge_agreement={agreement['agree']}/{agreement['n']}"
    if "unanswerable" in record:
        line += (f"  | answerable acc={fmt(record['answerable']['accuracy'])}"
                 f"  unanswerable acc={fmt(record['unanswerable']['accuracy'])}")
    print(line)


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 evaluation: hit rate + LLM-as-judge.")
    parser.add_argument("--limit", type=int, help="only use the first N questions of each file")
    parser.add_argument("--judge-only", type=Path, nargs="+", metavar="ANSWERS_JSON",
                        help="re-judge saved answers files instead of generating")
    parser.add_argument("--judge-model", default=JUDGE_LLM, help=f"Ollama model for the judge (default {JUDGE_LLM})")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    caller = LLMCaller()                 # generates the answers
    judge = LLMCaller(args.judge_model)  # grades them

    if args.judge_only:
        for path in args.judge_only:
            log_result(judge_only(path, judge))
        return

    hybrid = DenseRetriever.load(indexes_path=str(INDEXES / "hybrid"))
    # fixed = DenseRetriever.load(indexes_path=str(INDEXES / "fixed"))
    runs = {
        # "plain": lambda q: answer_plain(caller, q),
        "rag_hybrid": lambda q: answer_rag(caller, hybrid, q),
        # "rag_fixed": lambda q: answer_rag(caller, fixed, q),
        "rag_hybrid_city": lambda q: answer_rag(caller, hybrid, q, use_city=True),
    }

    questions = load_eval_set(DATA_FILES, args.limit)
    for name, system_fn in runs.items():
        print(f"{name} on {EVAL_SET} ({len(questions)} questions)")
        answers, retrieved = generate(system_fn, questions)
        save_json(answers, OUT / f"answers_{name}_{EVAL_SET}.json")
        log_result(evaluate_run(name, judge, questions, answers, retrieved))


if __name__ == "__main__":
    main()
