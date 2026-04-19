import time
from dataclasses import dataclass, field

from .prompts import PROMPTS, build_context


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: list
    latency_ms: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, embedder, vectorstore, llm, prompt_name="strict"):
        self.embedder = embedder
        self.vs = vectorstore
        self.llm = llm
        self.prompt_name = prompt_name
        self.prompt_template = PROMPTS[prompt_name]

    def answer(self, question, source_filter=None, k=4):
        t0 = time.perf_counter()
        where = {"source_type": source_filter} if source_filter else None
        chunks = self.vs.query(question, self.embedder, k=k, where=where)
        t1 = time.perf_counter()
        context = build_context(chunks, with_titles=self.prompt_name != "strict")
        prompt = self.prompt_template.format(context=context, question=question)
        ans = self.llm.generate(prompt)
        t2 = time.perf_counter()
        latency = {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        }
        return RAGResponse(question=question, answer=ans, retrieved_chunks=chunks, latency_ms=latency)
