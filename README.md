# ai-travel-assistant

> 🚧 **Work in progress** — this is a personal learning project, built step by step.

"Ask the Guidebook" is a question-answering assistant over a small set of [Wikivoyage](https://en.wikivoyage.org/) city guides. Ask it things like *"What's the cheapest way to get from Lisbon airport to the centre?"* and it answers using the guides, citing the passages it relied on, and says "I don't know" when the answer isn't there.

## Goals

- **Learn how LLM/RAG systems work from the inside**: ingestion, chunking, embeddings, retrieval and grounded generation are all built by hand, without frameworks like LangChain or LlamaIndex (those come at the very end, for comparison).
- **Measure, don't guess**: a hand-checked evaluation set is used to compare the RAG pipeline against a bare LLM and to judge every later change.
- **Keep it local and free**: the LLM runs through [Ollama](https://ollama.com/) and embeddings are computed locally with [fastembed](https://github.com/qdrant/fastembed).

## Status

| Stage | Description | Status |
|---|---|---|
| 0 | Setup (environment, first LLM call) | ✅ |
| 1 | Baseline and evaluation set | ✅ |
| 2 | Ingestion of Wikivoyage guides | ✅ |
| 3 | Chunking with metadata | ✅ |
| 4 | Embeddings and retrieval (hit rate@5) | ✅ |
| 5 | Grounded generation with citations | ✅ |
| 6 | Automated evaluation (LLM-as-judge) | ⏳ next |
| 7 | Improvements (hybrid search, reranking, metadata filtering, query rewriting) | ⏳ |
| 8 | Simple UI and rebuild with a RAG framework | ⏳ |

The full roadmap is in [`docs/project_guide.md`](docs/project_guide.md).

## Project structure

```
data/      downloaded guides, chunks and evaluation set
docs/      project guide and notes
scripts/   runnable pipeline steps
src/       core code
```

## Getting started

```bash
git clone https://github.com/luca-frittoli/ai-travel-assistant.git
cd ai-travel-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com/) installed with a model pulled locally. Configuration goes in a `.env` file (not committed).

## License

MIT. Guide content comes from Wikivoyage and is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
