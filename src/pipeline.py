from dataclasses import dataclass

from src import DenseRetriever, LLMCaller
from src.config import QUESTION_RULES
from src.utils import detect_city


@dataclass
class RAGResult:
    answer: str
    retrieved: list[tuple[float, dict]]
    city: str | None          # the city actually used to filter, if any

class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, caller: LLMCaller,
                 k: int = 3, detect_city_flag: bool = False):
        self.retriever = retriever
        self.caller = caller
        self.k = k
        self.detect_city_flag = detect_city_flag

    def answer(self, question: str, city: str | None = None) -> RAGResult:
        if city is None and self.detect_city_flag:
            city = detect_city(question)   # returns None if no single match
        retrieved = self.retriever.search(question, k=self.k, city=city)
        question_rules = QUESTION_RULES
        for _, chunk in retrieved:
            question_rules += f"{chunk['id']}: {chunk['text']}\n"
        answer = self.caller.ask_llm(question, system=question_rules)

        return RAGResult(
            answer=answer,
            retrieved=retrieved,
            city=city
        )