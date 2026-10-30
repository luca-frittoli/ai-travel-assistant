from ollama import chat
from src.config import DEFAULT_LLM


def ask_llm(prompt: str, model_name: str=DEFAULT_LLM, system: str=None) -> str:
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
    return chat(model=model_name, messages=messages).message.content