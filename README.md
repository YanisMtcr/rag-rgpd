# RAG Assistant DPO / GDPR

Deep Learning project, Centrale Lyon, 2025-2026.

A question answering system that answers French GDPR compliance questions by retrieving relevant passages from the GDPR text, CNIL practical guides, and CNIL sanctions.

Target audience: French speakers, therefore the prompts and generated answers are in French. The code and documentation are in English.

## Stack

- ChromaDB (vector database)
- sentence-transformers (embeddings)
- transformers (LLM, Apple Silicon MPS supported)
- gradio (interface)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download the PDFs listed in `data/raw/README.md` and drop them in `data/raw/`.
2. Run `notebooks/01_ingestion.ipynb` once to build the three ChromaDB collections.
3. Launch the interface:

```bash
python app.py
```

## Structure

```
src/                 pipeline modules
notebooks/           execution (01 ingestion, 02 experiments, 03 demo, 04 quick tests)
evaluation/          reference dataset and metrics
app.py               gradio interface
```

## Models compared

Three embedders, three LLMs, three prompt templates are compared in `02_experiments`.

Embedders:
- `sentence-transformers/all-MiniLM-L6-v2` (baseline, English)
- `intfloat/multilingual-e5-base` (multilingual)
- `OrdalieTech/Solon-embeddings-base-0.1` (French optimized)

LLMs (all run locally on Apple Silicon M3 Pro via MPS):
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (weak baseline)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct` (best performing locally)

Prompt templates: strict, citation (forces source citation), structured (three-part answer).
