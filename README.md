# RAG DPO / GDPR

Deep Learning project, Centrale Lyon, 2025-2026.

A question answering assistant over French GDPR content. The corpus is the GDPR text itself, about ten CNIL practical guides, and four CNIL sanction rulings (Google 2019 and 2025, an anonymised 2020 ruling, and France Travail 2026).

Demo video: [`Demo.mov`](Demo.mov).

Questions and answers are in French because that is what I care about for this use case. The code, filenames and comments are in English.

## Stack

- ChromaDB for the vector store, persisted under `data/chroma_db/`
- sentence-transformers for the embeddings
- transformers for the LLM, everything runs locally on a Mac M3 Pro through MPS (CUDA path is also handled in `generation.py`)
- gradio for the interface

## Repo layout

```
src/                  pipeline modules (ingestion, embedding, vectorstore, generation, prompts, rag_pipeline)
notebooks/            runnable notebooks (see below)
evaluation/           QA dataset and metrics
data/raw/             source PDFs (not versioned, see data/raw/README.md)
data/chroma_db/       ChromaDB persisted store
app.py                gradio interface
```

The notebooks:
- `01_ingestion` builds the three ChromaDB collections
- `02_experiments` compares embedders, LLMs and prompts
- `03_demo` launches the interface
- `04_tests` is a quick side by side of the three LLMs on three questions, before running the full benchmark

## What was compared

Three embedders:
- `sentence-transformers/all-MiniLM-L6-v2` (English baseline)
- `intfloat/multilingual-e5-base`
- `OrdalieTech/Solon-embeddings-base-0.1` (French oriented)

Three LLMs, all running locally:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small baseline)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Three prompt templates: strict, citation (forces a bracketed source), structured (three-block answer).

The config retained after `02_experiments` is hardcoded at the top of `app.py`.

## Install

```bash
pip install -r requirements.txt
```

The PDFs are not commited, see `data/raw/README.md` for the list. Once `data/raw/` is populated, `notebooks/01_ingestion.ipynb` fills the ChromaDB store, and `python app.py` opens the gradio interface on `localhost:7860`.

## Note on the demo

The demo video was recorded running everything locally on a Mac M3 Pro (MPS), so generation is slow (tens of seconds per answer). On a machine with a CUDA GPU, or behind a hosted API, latency would be much lower.
