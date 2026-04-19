from dataclasses import dataclass
from chromadb import PersistentClient


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


class VectorStore:
    def __init__(self, persist_path, collection_name):
        self.persist_path = str(persist_path)
        self.collection_name = collection_name
        self.client = PersistentClient(path=self.persist_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    def add_chunks(self, chunks, embedder):
        texts = [c.text for c in chunks]
        metas = [{k: v for k, v in c.metadata.items() if v is not None} for c in chunks]
        ids = [f"{m.get('source_file','x')}_{i}" for i, m in enumerate(metas)]
        embeddings = embedder.embed_passages(texts)
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

    def query(self, query_text, embedder, k=4, where=None):
        vec = embedder.embed_query(query_text)
        kw = {"query_embeddings": [vec], "n_results": k}
        if where:
            kw["where"] = where
        res = self.collection.query(**kw)
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for d, m, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(text=d, metadata=m, distance=dist))
        return out

    def count(self):
        return self.collection.count()
