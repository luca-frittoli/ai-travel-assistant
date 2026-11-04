from ollama import chat
from src.config import DEFAULT_LLM, JUDGE_RULES

class LLMCaller:
    def __init__(
            self,
            model_name: str=DEFAULT_LLM
    ):
        self.model_name = model_name

    def ask_llm(self, prompt: str, system: str = None) -> str:
        messages = [
            {
                'role': 'user',
                'content': prompt,
            },
        ]
        if system is not None:
            messages = [
                           {
                               'role': 'system',
                               'content': system
                           }
                       ] + messages
        return chat(model=self.model_name, messages=messages).message.content

    def judge(self, question: str, expected_answer: str, system_answer: str):
        prompt = ('Evaluate the following:'
                  '\nQuestion: ' + question +
                  '\nReference answer: ' + expected_answer +
                  '\nAssistant answer: ' + system_answer
                  )
        return self.ask_llm(
            prompt=prompt,
            system=JUDGE_RULES
        )
