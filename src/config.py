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

# Few-shot examples for the judge, aimed at the mistakes found while spot-checking:
# penalising reference details the question didn't ask for, rewarding refusals on
# answerable questions, being fooled by boilerplate. They use cities that are NOT in
# CITIES, so they can't leak eval answers into the judge (that would inflate agreement).
JUDGE_EXAMPLES = (
    'Examples:\n'
    '1. Question: What is the main square in Krakow called?\n'
    '   Reference: Rynek Glowny, one of the largest medieval squares in Europe, with the Cloth Hall in the middle.\n'
    '   Answer: It is called Rynek Glowny.\n'
    '   -> correct: the question only asks the name; size and Cloth Hall are not asked, so leaving them out is fine.\n'
    '2. Question: How long does the CAT train take from Vienna airport to the centre?\n'
    '   Reference: The CAT takes 16 minutes to Wien Mitte; the S7 is cheaper but slower.\n'
    '   Answer: Chunks used: Vienna-012. The CAT takes 16 minutes. Answer is not missing.\n'
    '   -> correct: the key fact (16 minutes) is there; ignore the boilerplate and the unasked S7 detail.\n'
    '3. Question: Do I need to validate my bus ticket in Prague?\n'
    '   Reference: Yes, validate it in the yellow machine when you board; inspectors fine you on the spot.\n'
    '   Answer: The information is missing from the provided chunks.\n'
    '   -> incorrect: the reference has an answer, so saying it is missing misses the main point.\n'
    '4. Question: Which metro line goes to the Vatican in Rome, and what is the nearest stop?\n'
    '   Reference: Line A, stop Ottaviano.\n'
    '   Answer: Take line A and get off at Spagna.\n'
    '   -> partial: the line is right, the stop is wrong.\n'
    '5. Question: Is there a metro from Krakow airport to the centre?\n'
    '   Reference: No, take the train to Krakow Glowny (about 20 minutes).\n'
    '   Answer: Yes, metro line 1 runs from the airport to the old town.\n'
    '   -> incorrect: it contradicts the main point (there is no metro).\n'
    '6. Question: How much is the Vienna City Card?\n'
    '   Reference: Not in the guide.\n'
    '   Answer: My sources do not say, but it usually costs around 17 euros.\n'
    '   -> incorrect: after saying it does not know, it still guesses a price.\n'
)

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
    'Ignore citation markers like [chunk 3] and boilerplate such as "Chunks used: ..." or '
    '"Answer is not missing": grade the facts the answer gives.\n'
    '\n'
    + JUDGE_EXAMPLES +
    '\n'
    'Reply in JSON with keys "reasoning" (one or two sentences) and "verdict" '
    '("correct", "partial" or "incorrect").'
)
