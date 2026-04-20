import re
from pathlib import Path
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _load_pdf_text(path):
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


def _split_fallback(text, size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


def parse_rgpd(path):
    text = _load_pdf_text(path)
    parts = re.split(r"(Article\s+\d+[a-z]?)", text)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        m = re.search(r"\d+", header)
        if not m:
            continue
        num = int(m.group(0))
        full = f"{header}\n{body}"
        meta_base = {
            "source_type": "rgpd",
            "article": num,
            "source_file": path.name,
            "title": f"RGPD {header}",
        }
        if len(full) > 2000:
            for sub in _split_fallback(full):
                chunks.append(Chunk(sub, dict(meta_base)))
        else:
            chunks.append(Chunk(full, meta_base))
    return chunks


def parse_fiche_cnil(path):
    text = _load_pdf_text(path)
    name = path.stem.replace("cnil_fiche_", "").replace("_", " ")
    chunks = []
    for sub in _split_fallback(text):
        chunks.append(Chunk(sub, {
            "source_type": "fiche_cnil",
            "source_file": path.name,
            "title": f"CNIL fiche {name}",
        }))
    return chunks


def parse_sanction_cnil(path):
    text = _load_pdf_text(path)
    entity = path.stem.replace("cnil_sanction_", "").split("_")[0]
    year_match = re.search(r"_(\d{4})", path.stem)
    year = year_match.group(1) if year_match else ""
    sections = re.split(r"\b(I{1,3}\.\s+[A-Z]{2,})", text)

    chunks = []
    if len(sections) > 1:
        for i in range(1, len(sections) - 1, 2):
            header = sections[i].strip()
            body = sections[i + 1].strip()
            full = f"{header}\n{body}"
            for sub in _split_fallback(full, size=1200, overlap=150):
                chunks.append(Chunk(sub, {
                    "source_type": "sanction",
                    "source_file": path.name,
                    "entity": entity,
                    "year": year,
                    "title": f"Sanction CNIL {entity} {year} - {header}",
                }))
    else:
        for sub in _split_fallback(text):
            chunks.append(Chunk(sub, {
                "source_type": "sanction",
                "source_file": path.name,
                "entity": entity,
                "year": year,
                "title": f"Sanction CNIL {entity} {year}",
            }))
    return chunks


def ingest_all(raw_dir):
    raw_dir = Path(raw_dir)
    all_chunks = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        if pdf.name.startswith("rgpd_"):
            all_chunks.extend(parse_rgpd(pdf))
        elif pdf.name.startswith("cnil_fiche_"):
            all_chunks.extend(parse_fiche_cnil(pdf))
        elif pdf.name.startswith("cnil_sanction_"):
            all_chunks.extend(parse_sanction_cnil(pdf))
        else:
            print(f"skipped (unknown prefix): {pdf.name}")
    return all_chunks
