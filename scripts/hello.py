from src.LLMCaller import LLMCaller

if __name__ == '__main__':
    caller = LLMCaller()
    response = caller.ask_llm("Why is the sky blue?")
    print(response)