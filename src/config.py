DEFAULT_LLM = 'llama3.2'
DEFAULT_EMBEDDING = 'BAAI/bge-small-en-v1.5'
CITIES = [
    "Lisbon",
    "Milan",
    "Munich",
    "Sofia",
    "Chisinau",
    "Bilbao",
    "Zurich",
    "Copenhagen",
    "Dublin",
    "Pamplona"
]
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SHORT_TAIL_TOL = 50

QUESTION_RULES = "keep the answer under 50 words\nanswer only from the following chunks\ncite which chunks were used\ntell if the answer is missing\nchunks:\n"

# The judge is a different, bigger model than the generator: a 3B model grading its
# own answers agreed with hand labels only 59% of the time (Stage 6 audit, 2026-11-12).
#
# mistral-small (24B, Q4) is a ~14 GB download and needs ~16 GB of free memory:
#   - Mac (Apple Silicon): 24 GB unified memory minimum, 32 GB+ comfortable
#   - NVIDIA GPU: 16 GB VRAM (or 12 GB with partial CPU offload, slower)
#   - CPU only: 32 GB RAM, but expect ~2-3 tokens/s (a full judge run takes hours)
# Lighter judges (less reliable, so check judge_agreement against data/hand_labels.json):
#   - 'qwen2.5:14b'  ~9 GB  -> 16 GB Mac / 12 GB GPU
#   - 'gemma3:12b'   ~8 GB  -> 16 GB Mac / 10 GB GPU
#   - 'qwen2.5:7b'   ~5 GB  -> 8-16 GB machines; noticeably weaker as a judge
# Avoid using DEFAULT_LLM as the judge: a model grading its own answers is too lenient.
# Override per run without editing this file: python -m scripts.evaluate --judge-model qwen2.5:14b
JUDGE_LLM = 'mistral-small'

JUDGE_RULES = (
    'You are grading the answer of a travel assistant against a reference answer.\n'
    'The reference answer is the ground truth. Do not use your own knowledge: if the reference '
    'and your knowledge disagree, the reference wins.\n'
    '\n'
    'Step 1 - unanswerable questions. If the reference answer is exactly "Not in the guide.":\n'
    '- "correct" only if the assistant says it does not know or the information is missing, '
    'without giving any answer.\n'
    '- "incorrect" if it attempts any answer, even a hedged guess, an estimate, or a note added '
    'after saying the information is missing.\n'
    'Stop here for these questions.\n'
    '\n'
    'Step 2 - answerable questions. The key facts are the parts of the reference that answer '
    'what the question asks. Other details in the reference (frequencies, extra prices, '
    'background) are not key facts: leaving them out is fine.\n'
    '- "correct": the answer states the key facts and does not contradict the reference. '
    'Extra details, wording differences and citations are fine.\n'
    '- "partial": the answer gets some key facts right but misses or gets wrong others.\n'
    '- "incorrect": the answer contradicts the reference on the main point, misses it, '
    'or says the information is missing.\n'
    '\n'
    'Only describe what the assistant answer actually says; quote it when in doubt.\n'
    'Ignore citation markers like [chunk 3] or "Chunks used: ...".\n'
    'Reply in JSON with keys "reasoning" (one or two sentences) and "verdict" '
    '("correct", "partial" or "incorrect").'
)
