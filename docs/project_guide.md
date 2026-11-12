# Project: "Ask the Guidebook", a travel Q&A assistant

Build a question-answering assistant over a small set of Wikivoyage city guides.
Example questions: "What's the cheapest way to get from Lisbon airport to the
centre?" or "Which Kyoto neighbourhoods are good for an evening walk?"

Why Wikivoyage: it's free and openly licensed, split into consistent sections
(Get in, Get around, See, Eat, Sleep), and contains enough specific detail that
a bare LLM will often get things wrong. That gives you something real to fix.

Rule for yourself: build everything by hand first. No LangChain or LlamaIndex
until the last stage, so you understand what those frameworks do for you.

------------------------------------------------------------------------------

## Stage 0: Setup

- Create a project with a virtual environment (uv or venv).
- Get an LLM answering a "hello" prompt from Python: a hosted API (Anthropic,
  OpenAI) or Ollama if you want everything local and free.
- Keep your API key in a `.env` file, load it with `python-dotenv`.
  Never hardcode it.

Done when: one script sends a prompt and prints the reply.

## Stage 1: Baseline and evaluation set

- Pick 10-15 cities.
- Write 15-20 questions whose answers you've personally checked in those guides.
- Store them in JSON or CSV: question, expected answer, city, section.
- Ask the bare LLM every question and record how it does.

This stage matters more than it looks: without it, you'll never know whether
your RAG system actually helps.

Hint: include a few questions whose answers are NOT in the guides. A good
system should say "I don't know" to those.

## Stage 2: Ingestion

- Download the guides as plain text. The MediaWiki API on wikivoyage.org can
  return clean extracts (search for "MediaWiki API extracts explaintext").
  `requests` is all you need.
- Save each guide to disk so you aren't re-downloading on every run.

Gotcha: look at the raw text before moving on. Leftover markup, empty sections
and listing templates will hurt you later.

## Stage 3: Chunking

- Split guides into chunks; keep metadata on each one (city, section heading,
  source URL).
- Start with section-based splitting, since the guides already have structure.
- Then try fixed-size chunks with overlap (e.g. 500 tokens, 50 overlap) and
  compare.

Think about: what happens to a very long "Eat" section? And to a 2-line
"Stay safe" section?

## Stage 4: Embeddings and retrieval

- Embed chunks with a local model via `sentence-transformers`
  (good starters: `all-MiniLM-L6-v2` or `bge-small-en`).
- Store embeddings in Chroma (easiest) or FAISS (lower level, very educational).
- Write a `search(query, k=5)` function and test it by eye on your eval
  questions BEFORE involving the LLM.

Done when: for most eval questions, the chunk containing the answer is in the
top 5. Measure this as "hit rate@5" using the city/section labels from Stage 1.

## Stage 5: Generation

- Put the retrieved chunks into a prompt. Ask the LLM to:
  - answer ONLY from them,
  - cite which chunk(s) it used,
  - say so when the answer isn't there.
- Rerun the eval set and compare against the Stage 1 baseline.

Think about: how to format chunks in the prompt so citations are unambiguous.

## Stage 6: Evaluation, properly

- Score answers automatically with "LLM-as-judge": a second prompt comparing
  each answer against your expected answer.
- Spot-check the judge's verdicts by hand; judges make mistakes too.
- Track two numbers from now on: retrieval hit rate and answer correctness.
  Every later change should move one of them.

## Stage 7: Improvements (pick any, measure each against the eval set)

- Hybrid search: combine embeddings with keyword search (`rank_bm25`).
  Helps with exact names like stations and dishes.
- Reranking: retrieve 20 candidates, rerank with a cross-encoder model.
- Metadata filtering: detect the city in the question, search only that
  city's chunks.
- Query rewriting: have the LLM rephrase vague questions before retrieval.

## Stage 8: Polish and compare

- Wrap the system in a small Streamlit or Gradio UI.
- Rebuild the pipeline in LlamaIndex or LangChain. Notice how much code
  disappears, and how much control you give up.

------------------------------------------------------------------------------

Checkpoints: when you finish a stage, or get stuck, bring what you've got
for a review or a hint (not the full solution).

## My notes

- LLM choice: local via Ollama. Generator `llama3.2` (3B); judge `mistral-small`
  (24B), with `qwen2.5:14b` as a faster alternative judge. Embeddings: `BAAI/bge-small-en-v1.5`
  (fastembed).
- Cities picked: Lisbon, Milan, Munich, Sofia, Chisinau, Bilbao, Zurich, Copenhagen,
  Dublin, Pamplona.
- Stage results:

### Stage 6 baseline (2026-11-12)

Setup: merged eval set (`data/sample_questions.json` + `data/sample_questions_claude.json`,
48 questions: 40 answerable, 8 "Not in the guide."), k=3, generator `llama3.2`,
judge `mistral-small` with few-shot rules. Score = (correct + 0.5 × partial) / n.

| System     | Hit rate | Score | Accuracy | Answerable score | Unanswerable score | Judge agreement |
|------------|----------|-------|----------|------------------|--------------------|-----------------|
| plain      | -        | 0.27  | 0.17     | 0.33             | 0.00               | 39/48           |
| rag_fixed  | 0.93     | 0.76  | 0.73     | 0.76             | 0.75               | 42/48           |
| **rag_hybrid** | **1.00** | **0.78** | 0.69 | 0.74          | 1.00               | 38/48           |

**Baseline for Stage 7: rag_hybrid, score 0.78, hit rate 1.00.** Every Stage 7
change gets compared to this. One question is about 2 points, so treat differences
under about 5 points as noise.

How the judge got here:
- The first judge was `llama3.2` grading its own answers. It agreed with my hand grades
  (`data/hand_labels.json`, 144 verdicts) only 59% of the time, and ranked plain above RAG.
- Changes: a separate, bigger judge; rewritten rules (the reference is the truth,
  unanswerable questions checked first, key facts = what the question asks); JSON schema
  output at temperature 0; judge errors count as incorrect; few-shot examples from cities
  outside the eval set.
- Agreement with the hand grades: 59% → 69% (mistral-small, new rules) → **83%**
  (+ few-shot); `qwen2.5:14b` reached 78%. 83% is optimistic, because the examples were
  written after seeing these disagreements. Hand-check about 15-20 fresh verdicts after
  the next generation run.
- Judge still leans slightly harsh: it gives partial/incorrect where I'd give
  correct/partial. My hand-graded scores: plain 0.38, rag_fixed 0.77, rag_hybrid 0.89.

Observations for Stage 7:
- Hybrid already has a perfect hit rate on this set, so most of the gain now has to
  come from generation.
- The answer prompt (`tell if the answer is missing`) makes answers include
  boilerplate like "Answer is missing" even when they do answer, and sometimes leads
  to real refusals on answerable questions. That's the first thing to fix.
- Plain llama3.2 guesses on every "Not in the guide." question (0/8); hybrid refuses
  all 8 correctly.
- A Copenhagen chunk was retrieved for a Munich question → metadata filtering by city.
