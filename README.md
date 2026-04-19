# RAG Assistant DPO / RGPD

Projet deep learning, Centrale Lyon, 2025-2026.

Un systeme de question reponse qui s'appuie sur le RGPD, les fiches pratiques CNIL et les sanctions CNIL pour repondre a des questions de conformite.

## Stack

- ChromaDB (base vectorielle)
- sentence-transformers (embeddings)
- transformers + bitsandbytes (LLM quantise 4-bit)
- gradio (interface)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Telecharger les PDFs listes dans `data/raw/README.md` et les placer dans `data/raw/`.
2. Lancer `notebooks/01_ingestion.ipynb` pour creer les collections ChromaDB (une fois).
3. Lancer l'interface:

```bash
python app.py
```

## Structure

```
src/                 modules du pipeline
notebooks/           execution (01 ingestion, 02 experiences, 03 demo)
evaluation/          dataset de ref + metriques
app.py               interface gradio
```
