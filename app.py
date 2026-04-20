import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.embedding import Embedder
from src.vectorstore import VectorStore
from src.generation import LLMGenerator
from src.rag_pipeline import RAGPipeline


EMBED_MODEL = "OrdalieTech/Solon-embeddings-base-0.1"
COLLECTION = "collection_solon_base"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT = "citation"


print("loading embedder...")
embedder = Embedder(EMBED_MODEL)
print("loading vectorstore...")
vs = VectorStore("data/chroma_db", COLLECTION)
print("loading llm (1-2 min)...")
llm = LLMGenerator(LLM_MODEL)
pipe = RAGPipeline(embedder, vs, llm, prompt_name=PROMPT)
print("ready")


def format_sources(chunks):
    if not chunks:
        return "_pas de sources retournees_"
    lines = ["**Sources utilisees:**"]
    for c in chunks:
        title = c.metadata.get("title") or c.metadata.get("source_file", "?")
        lines.append(f"- {title}")
    return "\n".join(lines)


def respond(message, history, source_filter, k):
    where = None if source_filter == "tout" else source_filter
    resp = pipe.answer(message, source_filter=where, k=int(k))
    srcs = format_sources(resp.retrieved_chunks)
    latency = f"_retrieval {resp.latency_ms['retrieval']}ms / generation {resp.latency_ms['generation']}ms_"
    return f"{resp.answer}\n\n---\n{srcs}\n\n{latency}"


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Radio(
            choices=["tout", "rgpd", "fiche_cnil", "sanction"],
            value="tout",
            label="Filtrer les sources",
        ),
        gr.Slider(1, 8, value=4, step=1, label="Nb chunks (top-k)"),
    ],
    title="Assistant DPO - RAG RGPD",
    description="Pose une question sur le RGPD, les pratiques CNIL ou les sanctions.",
    examples=[
        "Dois-je nommer un DPO si j'ai 30 salaries ?",
        "Quelle est la duree de conservation max d'un CV ?",
        "Pourquoi Google a ete sanctionne en 2019 ?",
        "Que faire en cas de fuite de donnees ?",
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
