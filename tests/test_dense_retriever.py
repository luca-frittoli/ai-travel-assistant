import numpy as np
import pytest

from src.DenseRetriever import DenseRetriever


@pytest.fixture
def retriever(fake_embedding, chunks, tmp_path):
    r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path / "idx"))
    r.create_embedding_matrix()
    return r


class TestTextChunks:
    def test_extracts_text_field_in_order(self, fake_embedding, chunks, tmp_path):
        r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path))
        assert r.text_chunks == [c["text"] for c in chunks]

    def test_custom_text_keyword(self, fake_embedding, chunks, tmp_path):
        r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path))
        r.extract_text_from_chunks(text_keyword="id")
        assert r.text_chunks == [c["id"] for c in chunks]


class TestEmbeddingMatrix:
    def test_one_row_per_chunk(self, retriever, chunks):
        assert retriever.embeddings.shape[0] == len(chunks)

    def test_rows_are_normalised(self, retriever):
        norms = np.linalg.norm(retriever.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestSearch:
    def test_raises_without_embeddings(self, fake_embedding, chunks, tmp_path):
        r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path))
        with pytest.raises(RuntimeError):
            r.search("airport")

    def test_returns_k_score_chunk_pairs(self, retriever):
        results = retriever.search("airport metro", k=2)
        assert len(results) == 2
        for score, chunk in results:
            assert isinstance(score, float)
            assert isinstance(chunk, dict) and "id" in chunk

    def test_k_larger_than_corpus_returns_everything(self, retriever, chunks):
        assert len(retriever.search("anything", k=100)) == len(chunks)

    def test_scores_sorted_descending(self, retriever):
        scores = [s for s, _ in retriever.search("beer gardens airport", k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_most_similar_chunk_ranks_first(self, retriever):
        _, top = retriever.search("where to drink beer in beer halls", k=1)[0]
        assert top["id"] == "Munich-001"

    def test_score_is_dot_product(self, retriever):
        score, chunk = retriever.search("pastel de nata", k=1)[0]
        i = retriever.chunks.index(chunk)
        q = retriever._embed_query("pastel de nata")
        assert score == pytest.approx(float(retriever.embeddings[i] @ q))


class TestPersistence:
    def test_save_raises_without_embeddings(self, fake_embedding, chunks, tmp_path):
        r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path))
        with pytest.raises(RuntimeError):
            r.save()

    def test_save_creates_nested_dir_and_files(self, retriever):
        retriever.save()
        assert retriever._chunk_file_name.exists()
        assert retriever._embedding_file_name.exists()

    def test_embedding_file_name_has_no_slash_from_model_name(self, retriever):
        # default model name "BAAI/bge-small-en-v1.5" must not create a sub-directory
        assert retriever._embedding_file_name.parent == retriever.indexes_path
        assert retriever._embedding_file_name.name == "embeddings_BAAI_bge-small-en-v1.5.npz"

    def test_save_load_roundtrip(self, retriever, chunks):
        retriever.save()
        loaded = DenseRetriever.load(
            indexes_path=str(retriever.indexes_path),
            model_name=retriever.embedding_model.model_name,
        )
        assert loaded.chunks == chunks
        np.testing.assert_allclose(loaded.embeddings, retriever.embeddings)
        assert [c["id"] for _, c in loaded.search("airport", k=4)] == \
               [c["id"] for _, c in retriever.search("airport", k=4)]

    def test_load_rejects_mismatched_ids(self, retriever):
        retriever.save()
        data = np.load(retriever._embedding_file_name)
        np.savez(
            retriever._embedding_file_name,
            embeddings=data["embeddings"],
            ids=data["ids"][::-1],
        )
        with pytest.raises(ValueError):
            DenseRetriever.load(
                indexes_path=str(retriever.indexes_path),
                model_name=retriever.embedding_model.model_name,
            )

    def test_load_missing_index_raises(self, fake_embedding, tmp_path):
        with pytest.raises(FileNotFoundError):
            DenseRetriever.load(indexes_path=str(tmp_path / "nope"))


@pytest.mark.integration
def test_real_model_finds_right_city(tmp_path, chunks):
    """Uses the real bge-small model (downloads it if needed)."""
    r = DenseRetriever(chunks=chunks, indexes_path=str(tmp_path))
    r.create_embedding_matrix()
    _, top = r.search("How long is the bus from Dublin airport?", k=1)[0]
    assert top["id"] == "Dublin-001"
