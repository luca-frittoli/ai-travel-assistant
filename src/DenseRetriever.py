import json
import numpy as np
from pathlib import Path
from src.config import DEFAULT_EMBEDDING
from fastembed import TextEmbedding

class DenseRetriever:
    def __init__(
            self,
            chunks: list[dict],
            model_name: str=DEFAULT_EMBEDDING,
            indexes_path: str="indexes/hybrid"
    ):
        self.chunks = chunks
        self.embedding_model = TextEmbedding(model_name)
        self.indexes_path = Path(indexes_path)
        self.embeddings = None
        self._text_chunks = None
        self._embedding_file_name = self.indexes_path / f"embeddings_{self.embedding_model.model_name.replace('/', '_')}.npz"
        self._chunk_file_name = self.indexes_path / "chunks.json"
        self.chunk_cities = np.array([c['id'].split('-')[0].lower() for c in self.chunks])

    def extract_text_from_chunks(self, text_keyword: str="text"):
        self._text_chunks = [chunk[text_keyword] for chunk in self.chunks]

    @property
    def text_chunks(self) -> list[str]:
        if self._text_chunks is None:
            self.extract_text_from_chunks()
        return self._text_chunks

    def create_embedding_matrix(self):
        self.embeddings = np.array(list(self.embedding_model.embed(self.text_chunks)))

    @classmethod
    def load(cls, indexes_path: str="indexes/hybrid", model_name: str=DEFAULT_EMBEDDING):

        chunk_file_name = Path(indexes_path) / "chunks.json"
        with open(chunk_file_name, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        chunk_ids = [chunk["id"] for chunk in chunks]

        retriever = cls(chunks=chunks, model_name=model_name, indexes_path=indexes_path)
        data = np.load(retriever._embedding_file_name)
        retriever.embeddings = data["embeddings"]  # shape (n_chunks, 384)
        saved_ids = data["ids"]  # e.g. ["Lisbon-000", "Lisbon-001", ...]

        if list(saved_ids) != chunk_ids:
            raise ValueError("Embeddings don't match the chunks; rebuild the index.")

        return retriever

    def save(self):
        if self.embeddings is None:
            raise RuntimeError("Embeddings are missing, call create_embedding_matrix() or load() first")

        self.indexes_path.mkdir(parents=True, exist_ok=True)

        with open(self._chunk_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        np.savez(
            self._embedding_file_name,
            embeddings=self.embeddings,
            ids=np.array([chunk["id"] for chunk in self.chunks]),
        )

    def _embed_query(self, query: str):
        return np.array(list(self.embedding_model.embed([query])))[0]

    def search(self, query: str, k: int = 5, city: str | None=None) -> list[tuple[float, dict]]:
        if self.embeddings is None:
            raise RuntimeError("Embeddings are missing, call create_embedding_matrix() or load() first")

        query_embedding = self._embed_query(query)
        if city:
            city_embeddings = self.embeddings[np.where(self.chunk_cities == city.lower())]
            similarity_scores = city_embeddings @ query_embedding
        else:
            similarity_scores = self.embeddings @ query_embedding
        top_k = np.argsort(similarity_scores)[::-1][:k]
        return [
            (float(similarity_scores[i]), self.chunks[i])
            for i in top_k
        ]


