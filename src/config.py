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

JUDGE_RULES = ('You are grading the answer of a travel assistant against a reference answer.\n'
               'Rules:\n'
               '- Compare the assistant answer to the reference answer.\n'
               '- "correct": the answer contains the key facts of the reference answer and does not contradict it. Extra correct details are fine.\n'
               '- "partial": the answer gets some key facts right but misses or gets wrong others.\n'
               '- "incorrect": the answer contradicts the reference or misses its main point.\n'
               '- If the reference answer is "Not in the guide.", the answer is "correct" only if the assistant clearly says it does not know. Any attempted answer, even a hedged guess, is "incorrect".\n'
               '- Ignore citation markers like [chunk 3].\n'
               '- Answer in json format with keys "reasoning": "...", "verdict": "correct" | "partial" | "incorrect"'
               )