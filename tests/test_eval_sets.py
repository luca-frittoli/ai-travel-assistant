"""Sanity checks on the evaluation sets: a wrong label silently skews hit rate."""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from src.config import CITIES
from src.evaluation import UNANSWERABLE

ROOT = Path(__file__).parent.parent
EVAL_FILES = sorted((ROOT / "data").glob("sample_questions*.json"))
INDEX_CHUNKS = ROOT / "indexes" / "hybrid" / "chunks.json"
KEYS = {"question", "expected_answer", "city", "section"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sections_by_city():
    if not INDEX_CHUNKS.exists():
        pytest.skip("indexes/hybrid not built (run scripts.embedding)")
    out = defaultdict(set)
    for c in load(INDEX_CHUNKS):
        out[c["city"]].update(c["sections"])
    return out


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
class TestEvalSet:
    def test_schema(self, path):
        for q in load(path):
            assert set(q) == KEYS, q
            assert q["question"].strip() and q["expected_answer"].strip(), q
            assert q["city"] in CITIES, q

    def test_unanswerable_iff_no_section(self, path):
        for q in load(path):
            assert (q["section"] is None) == (q["expected_answer"] == UNANSWERABLE), q

    def test_no_duplicate_questions(self, path):
        questions = [q["question"].strip().lower() for q in load(path)]
        assert len(questions) == len(set(questions))

    def test_sections_exist_in_index(self, path, sections_by_city):
        for q in load(path):
            if q["section"] is not None:
                assert q["section"] in sections_by_city[q["city"]], q
