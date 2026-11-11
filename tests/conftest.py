import re
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

import src.DenseRetriever as dense_retriever_module
import src.LLMCaller as llm_caller_module

DIM = 64


class FakeEmbedding:
    """Stand-in for fastembed.TextEmbedding: deterministic bag-of-words vectors.

    Each word is hashed to a dimension, and the vector is L2-normalised (like bge),
    so texts that share words get a higher dot product. No model download needed.
    """

    def __init__(self, model_name: str = "fake/bag-of-words", **kwargs):
        self.model_name = model_name
        self.calls = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        for text in texts:
            vec = np.zeros(DIM, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            norm = np.linalg.norm(vec)
            yield vec / norm if norm else vec


@pytest.fixture
def fake_embedding(monkeypatch):
    monkeypatch.setattr(dense_retriever_module, "TextEmbedding", FakeEmbedding)
    return FakeEmbedding


@pytest.fixture
def chunks():
    return [
        {"id": "Lisbon-001", "city": "Lisbon", "sections": ["Get in"],
         "text": "Lisbon>Get in\nThe metro red line connects the airport to the city."},
        {"id": "Lisbon-002", "city": "Lisbon", "sections": ["Eat"],
         "text": "Lisbon>Eat\nTry pastel de nata, a custard tart."},
        {"id": "Munich-001", "city": "Munich", "sections": ["Drink"],
         "text": "Munich>Drink\nBeer gardens and beer halls serve beer all day."},
        {"id": "Dublin-001", "city": "Dublin", "sections": ["Get in"],
         "text": "Dublin>Get in\nLocal buses from the airport take up to one hour."},
    ]


class FakeChat:
    """Records calls to ollama.chat and returns a canned reply."""

    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


@pytest.fixture
def fake_chat(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(llm_caller_module, "chat", chat)
    return chat


def ollama_available(model: str) -> bool:
    try:
        import ollama
        names = {m.model for m in ollama.list().models}
        return model in names or f"{model}:latest" in names
    except Exception:
        return False
