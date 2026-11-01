import json
from src.utils import ask_llm
from pathlib import Path
from src.DenseRetriever import DenseRetriever

data_path = 'data/sample_questions.json'
out_path = Path('out')
indexes_path = Path('indexes')
method = 'hybrid'

rules = "keep the answer under 50 words\nanswer only from the following chunks\ncite which chunks were used\ntell if the answer is missing\nchunks:\n"

if __name__ == '__main__':
    with open(data_path, 'rb') as f:
        questions = json.load(f)

    retriever = DenseRetriever.load(indexes_path=str(indexes_path / method))

    answers = []
    n = len(questions)
    for i, question in enumerate(questions):
        print(f"Answering question {i+1}/{n}...")
        closest_chunks = retriever.search(query=question['question'], k=3)

        question_rules = ""
        for _, chunk in closest_chunks:
            question_rules += f"{chunk['id']}: {chunk['text']}\n"

        response = ask_llm(question['question'], system=rules+question_rules)
        answers.append({
            'question': question['question'],
            'expected_answer': question['expected_answer'],
            'model_answer': response
        })

    with open(out_path / f'out_{method}_{retriever.embedding_model.model_name.replace("/", "_")}.json', 'w') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
