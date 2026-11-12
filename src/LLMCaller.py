from ollama import chat
from src.config import DEFAULT_LLM, JUDGE_RULES
from src.evaluation import VERDICTS

# Structured output: Ollama constrains the judge's reply to this JSON schema, so it
# can't forget a brace or return "reasoning" as a list. "reasoning" comes first so the
# model thinks before it commits to a verdict.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reasoning", "verdict"],
}

class LLMCaller:
    def __init__(
            self,
            model_name: str=DEFAULT_LLM
    ):
        self.model_name = model_name

    def ask_llm(self, prompt: str, system: str = None, format: dict | None = None,
                options: dict | None = None) -> str:
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
        return chat(model=self.model_name, messages=messages, format=format, options=options).message.content

    def judge(self, question: str, expected_answer: str, system_answer: str):
        prompt = ('Evaluate the following:'
                  '\nQuestion: ' + question +
                  '\nReference answer: ' + expected_answer +
                  '\nAssistant answer: ' + system_answer
                  )
        return self.ask_llm(
            prompt=prompt,
            system=JUDGE_RULES,
            format=VERDICT_SCHEMA,
            options={"temperature": 0},
        )
