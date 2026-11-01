from ollama import chat
from src.config import DEFAULT_LLM, CHUNK_SIZE, CHUNK_OVERLAP, SHORT_TAIL_TOL


def ask_llm(prompt: str, model_name: str=DEFAULT_LLM, system: str=None) -> str:
    messages = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]
    if system is not None:
        messages = [
            {
                'role': 'system',
                'content': system
            }
        ] + messages
    return chat(model=model_name, messages=messages).message.content

def chunk_words(
        extract: str, chunk_size: int=CHUNK_SIZE, chunk_overlap: int=CHUNK_OVERLAP, short_tail_tol: int=SHORT_TAIL_TOL
) -> list[str]:
    assert chunk_overlap < chunk_size, "chunk size must be greater than chunk overlap"
    out = []
    split_extract = extract.split()
    start = 0
    while start + chunk_size < len(split_extract):
        out.append(' '.join(split_extract[start:start+chunk_size]))
        start = start + chunk_size - chunk_overlap

    if len(split_extract) - start <= short_tail_tol:
        if len(out) > 0:
            out[-1] += ' ' + ' '.join(split_extract[start+chunk_overlap:])
        else:
            out.append(' ' + ' '.join(split_extract[start+chunk_overlap:]))
    else:
        out.append(' '.join(split_extract[start:]))
    return out
