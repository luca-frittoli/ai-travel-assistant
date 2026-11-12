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
| 6 | Automated evaluation (LLM-as-judge) | ✅ |
| 7 | Improvements (hybrid search, reranking, metadata filtering, query rewriting) | ⏳ next |
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

### Models

Two local models are used, both set in [`src/config.py`](src/config.py):

| Role | Default | Download | Free memory needed |
|---|---|---|---|
| Generator (answers questions) | `llama3.2` (3B) | ~2 GB | ~4 GB, runs almost anywhere |
| Judge (grades answers in the evaluation) | `mistral-small` (24B) | ~14 GB | ~16 GB |

```bash
ollama pull llama3.2
ollama pull mistral-small   # only needed for the evaluation
```

Minimum specs for the `mistral-small` judge: a Mac with **24 GB** unified memory (32 GB+ is comfortable), an NVIDIA GPU with **16 GB** VRAM, or CPU-only with 32 GB RAM (works, but a full evaluation takes hours).

On a smaller machine, pick a lighter judge with `--judge-model`:

| Judge | Download | Fits on |
|---|---|---|
| `qwen2.5:14b` | ~9 GB | 16 GB Mac / 12 GB GPU |
| `gemma3:12b` | ~8 GB | 16 GB Mac / 10 GB GPU |
| `qwen2.5:7b` | ~5 GB | 8–16 GB machines (noticeably weaker) |

```bash
python -m scripts.evaluate --judge-model qwen2.5:14b
```

Smaller judges make more mistakes, so check the `judge_agreement` number the evaluation prints: it compares the judge's verdicts with hand-graded ones in `data/hand_labels.json`. Don't use the generator model as its own judge: it grades its own answers too generously.

## License

MIT. Guide content comes from Wikivoyage and is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
