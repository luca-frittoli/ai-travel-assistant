"""Tests for the pure helpers in scripts/evaluate.py (no Ollama, no indexes needed)."""
import json

from scripts.evaluate import judged_path, load_eval_set


def write(tmp_path, name, questions):
    path = tmp_path / name
    path.write_text(json.dumps(questions), encoding="utf-8")
    return str(path)


class TestLoadEvalSet:
    def test_concatenates_files_in_order(self, tmp_path):
        a = write(tmp_path, "a.json", [{"question": "A1"}, {"question": "A2"}])
        b = write(tmp_path, "b.json", [{"question": "B1"}])
        assert [q["question"] for q in load_eval_set([a, b])] == ["A1", "A2", "B1"]

    def test_limit_applies_to_each_file(self, tmp_path):
        # --limit 1 is a quick smoke test: it should still touch every file.
        a = write(tmp_path, "a.json", [{"question": "A1"}, {"question": "A2"}])
        b = write(tmp_path, "b.json", [{"question": "B1"}, {"question": "B2"}])
        assert [q["question"] for q in load_eval_set([a, b], limit=1)] == ["A1", "B1"]


class TestJudgedPath:
    def test_includes_run_and_judge_model(self, tmp_path):
        assert judged_path("rag_hybrid_eval", "mistral-small", out=tmp_path) == \
            tmp_path / "judged_rag_hybrid_eval__mistral-small.json"

    def test_model_tag_is_filename_safe(self, tmp_path):
        assert judged_path("plain_eval", "qwen2.5:14b", out=tmp_path).name == "judged_plain_eval__qwen2.5-14b.json"
