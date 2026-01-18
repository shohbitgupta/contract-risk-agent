from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    """
    Deterministic embedding generator for legal text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()