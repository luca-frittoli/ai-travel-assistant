import json
import numpy as np
from pathlib import Path
from src.DenseRetriever import DenseRetriever

data_path = 'data/sample_questions.json'
out_path = Path('out')
indexes_path = Path('indexes')
verbose = False

if __name__ == '__main__':
    with open(data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    for method in ['hybrid', 'fixed']:
        print(f"----------------------------\nMethod: {method}\n")
        retriever = DenseRetriever.load(indexes_path=str(indexes_path / method))

        hits = []

        for question in questions:
            if question["section"] is not None:
                if verbose:
                    print("Question: " + question['question'] + "\nClosest chunks:\n")
                closest_chunks = retriever.search(query=question['question'], k=5)
                hit = 0
                for score, chunk in closest_chunks:
                    if chunk["city"] == question["city"] and question["section"] in chunk['sections']:
                        hit = 1
                    if verbose:
                        print(f"Score: {score:.4f}, hit: {hit}\n" + chunk["text"] + '\n')
                hits.append(hit)

        print(f"Hit rate top 5: {np.mean(hits):.4f}\n")