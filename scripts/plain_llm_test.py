import json
from src.LLMCaller import LLMCaller
data_path = 'data/sample_questions.json'
out_path = 'out/out.json'

if __name__ == '__main__':
    with open(data_path, 'rb') as f:
        questions = json.load(f)

    caller = LLMCaller
    answers = []
    for question in questions:
        response = caller.ask_llm(question['question'], system="keep the answer under 50 words")
        answers.append({
            'question': question['question'],
            'expected_answer': question['expected_answer'],
            'model_answer': response
        })

    with open(out_path, 'w') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
