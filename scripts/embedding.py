import json
from pathlib import Path
from src.config import CITIES
from src.DenseRetriever import DenseRetriever

guides_path = Path('guides')

if __name__ == '__main__':
    for method in ['hybrid', 'fixed']:
        all_chunks = []
        for city in CITIES:
            with open(guides_path / f'{city}_chunked_{method}.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            all_chunks += chunks

        retriever = DenseRetriever(
            chunks=all_chunks,
            indexes_path=f"indexes/{method}"
        )

        retriever.create_embedding_matrix()
        retriever.save()