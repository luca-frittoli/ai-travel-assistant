import re

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SHORT_TAIL_TOL
from src.static.city_aliases import CITY_ALTERNATIVE_NAMES


_CITY_PATTERNS = {
    city: re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")\b"
    )
    for city, aliases in CITY_ALTERNATIVE_NAMES.items()
}


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

def find_cities(text: str) -> list[str]:
    """All cities mentioned in the text, in CITIES order."""
    text = text.lower()
    return [city for city, pattern in _CITY_PATTERNS.items() if pattern.search(text)]


def detect_city(text: str) -> str | None:
    """The city to filter on: only if exactly one city is mentioned, else None."""
    cities = find_cities(text)
    return cities[0] if len(cities) == 1 else None