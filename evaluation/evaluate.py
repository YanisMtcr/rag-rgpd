import json
import re
from statistics import mean, median


def load_dataset(path="evaluation/qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_recall(pipeline, dataset, k=4):
    hits = 0
    total = 0
    for q in dataset:
        expected = q.get("expected_sources", [])
        if not expected:
            continue
        total += 1
        res = pipeline.vs.query(q["question"], pipeline.embedder, k=k)
        retrieved_files = {c.metadata.get("source_file") for c in res}
        if any(src in retrieved_files for src in expected):
            hits += 1
    return hits / total if total else 0.0


def keyword_score(pipeline, dataset, k=4):
    scores = []
    details = []
    for q in dataset:
        if not q.get("expected_keywords"):
            continue
        resp = pipeline.answer(q["question"], k=k)
        answer_low = resp.answer.lower()
        found = sum(1 for kw in q["expected_keywords"] if kw.lower() in answer_low)
        s = found / len(q["expected_keywords"])
        scores.append(s)
        details.append({"id": q["id"], "score": s, "answer": resp.answer[:200]})
    return (mean(scores) if scores else 0.0), details


def judge_with_llm(judge_llm, pipeline, dataset, k=4):
    scores = []
    for q in dataset:
        resp = pipeline.answer(q["question"], k=k)
        prompt = (
            f"Question: {q['question']}\n"
            f"Reponse attendue: {q.get('reference_answer','')}\n"
            f"Reponse generee: {resp.answer}\n"
            "Note la reponse generee de 1 a 5 (1=faux, 3=partiellement correct, 5=equivalent).\n"
            "Reponds uniquement par un chiffre entre 1 et 5."
        )
        out = judge_llm.generate(prompt, max_new_tokens=5, temperature=0.0)
        m = re.search(r"[1-5]", out)
        if m:
            scores.append(int(m.group(0)))
    return mean(scores) if scores else 0.0


def measure_latency(pipeline, dataset, k=4):
    rets, gens, tots = [], [], []
    for q in dataset:
        resp = pipeline.answer(q["question"], k=k)
        rets.append(resp.latency_ms["retrieval"])
        gens.append(resp.latency_ms["generation"])
        tots.append(resp.latency_ms["total"])
    return {
        "retrieval": {"mean": mean(rets), "median": median(rets)},
        "generation": {"mean": mean(gens), "median": median(gens)},
        "total": {"mean": mean(tots), "median": median(tots)},
    }
