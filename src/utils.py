from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SHORT_TAIL_TOL


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
