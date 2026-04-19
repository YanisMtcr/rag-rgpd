from sentence_transformers import SentenceTransformer


E5_MODELS = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"}


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._is_e5 = model_name in E5_MODELS

    def embed_query(self, text):
        t = f"query: {text}" if self._is_e5 else text
        return self.model.encode([t], normalize_embeddings=True).tolist()[0]

    def embed_passages(self, texts):
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()
