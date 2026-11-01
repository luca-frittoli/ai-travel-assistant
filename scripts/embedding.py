import json
import numpy as np
from fastembed import TextEmbedding
from src.config import DEFAULT_EMBEDDING, CITIES

guides_path = 'guides/'

if __name__ == '__main__':
    model = TextEmbedding(DEFAULT_EMBEDDING)
    all_chunks = []
    for city in CITIES:
        with open(f'{guides_path}{city}_chunked_hybrid.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        all_chunks += [chunk["text"] for chunk in chunks]
    embeddings = np.array(list(model.embed(all_chunks)))
    print(embeddings.shape)
    print(len(all_chunks)==len(embeddings))
    print(min([len(e) for e in embeddings]), max([len(e) for e in embeddings]))

    query = "Is it possible to visit Copenhagen by bike?"
    q_embedding = np.array(list(model.embed([query])))[0]
    print(q_embedding.shape)

    similarity_scores = embeddings @ q_embedding
    top_5 = np.argsort(similarity_scores)[::-1][:5]

    for i in top_5:
        print(f'{i+1}) Similarity: {similarity_scores[i]} --> {all_chunks[i]}\n')