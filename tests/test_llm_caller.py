import pytest

from src.config import DEFAULT_LLM, JUDGE_RULES
from src.LLMCaller import LLMCaller
from tests.conftest import ollama_available


class TestAskLLM:
    def test_default_model(self):
        assert LLMCaller().model_name == DEFAULT_LLM

    def test_uses_given_model(self, fake_chat):
        LLMCaller(model_name="other-model").ask_llm("hi")
        assert fake_chat.calls[0]["model"] == "other-model"

    def test_returns_message_content(self, fake_chat):
        fake_chat.reply = "Take the red line."
        assert LLMCaller().ask_llm("hi") == "Take the red line."

    def test_without_system_sends_only_user_message(self, fake_chat):
        LLMCaller().ask_llm("hello")
        assert fake_chat.calls[0]["messages"] == [{"role": "user", "content": "hello"}]

    def test_system_message_comes_first(self, fake_chat):
        LLMCaller().ask_llm("hello", system="be brief")
        assert fake_chat.calls[0]["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ]

    def test_one_call_per_question(self, fake_chat):
        caller = LLMCaller()
        caller.ask_llm("a")
        caller.ask_llm("b")
        assert len(fake_chat.calls) == 2


class TestJudge:
    def test_uses_judge_rules_as_system(self, fake_chat):
        LLMCaller().judge("Q?", "ref", "ans")
        system = fake_chat.calls[0]["messages"][0]
        assert system == {"role": "system", "content": JUDGE_RULES}

    def test_prompt_contains_question_reference_and_answer(self, fake_chat):
        LLMCaller().judge("Where is Tivoli?", "City centre.", "Near the station.")
        prompt = fake_chat.calls[0]["messages"][-1]["content"]
        assert "Where is Tivoli?" in prompt
        assert "City centre." in prompt
        assert "Near the station." in prompt

    def test_reference_and_answer_are_labelled(self, fake_chat):
        # The judge must be able to tell which text is the reference and which is the answer.
        LLMCaller().judge("Q", "REF_TEXT", "ANS_TEXT")
        prompt = fake_chat.calls[0]["messages"][-1]["content"]
        assert prompt.index("Reference answer") < prompt.index("REF_TEXT")
        assert prompt.index("Assistant answer") < prompt.index("ANS_TEXT")

    def test_returns_raw_model_output(self, fake_chat):
        fake_chat.reply = '{"reasoning": "ok", "verdict": "correct"}'
        assert LLMCaller().judge("Q", "R", "A") == fake_chat.reply


@pytest.mark.integration
@pytest.mark.skipif(not ollama_available(DEFAULT_LLM), reason="Ollama server/model not available")
def test_real_llm_answers():
    answer = LLMCaller().ask_llm("Reply with the single word: pong")
    assert "pong" in answer.lower()
