import numpy as np
from src.config import DEFAULT_EMBEDDING
from fastembed import TextEmbedding

class DenseRetriever:
    def __init__(
            self,
            chunks: list[dict],
            model_name: str=DEFAULT_EMBEDDING
    ):
        self.chunks = chunks
        self.embedding_model = TextEmbedding(model_name)
        self.embeddings = None
        self._text_chunks = None

    def extract_text_from_chunks(self, text_keyword: str="text"):
        self._text_chunks = [chunk[text_keyword] for chunk in self.chunks]

    @property
    def text_chunks(self):
        if self._text_chunks is None:
            self.extract_text_from_chunks()
        return self._text_chunks

    def create_embedding_matrix(self):
        self.embeddings = np.array(list(self.embedding_model.embed(self.text_chunks)))

    def load_embedding_matrix(self):
        pass

    def save_embedding_matrix(self):
        pass

    def _embed_query(self, query: str):
        return np.array(list(self.embedding_model.embed([query])))[0]

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        assert self.embeddings is not None, "call create_embedding_matrix() or load() first"
        query_embedding = self._embed_query(query)
        similarity_scores = self.embeddings @ query_embedding
        top_k = np.argsort(similarity_scores)[::-1][:k]
        return [
            (float(similarity_scores[i]), self.chunks[i])
            for i in top_k
        ]


