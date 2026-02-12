from typing import List
from sentence_transformers import CrossEncoder
from vector_index.index_base import IndexDocument


class CrossEncoderReRankingAgent:
    """
    Cross-encoder reranker for improving semantic precision.
    Re-ranks vector search results using query-document scoring.
    """

    def __init__(self, model_name: str, top_k: int = 5):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        documents: List[IndexDocument],
    ) -> List[IndexDocument]:

        if not documents:
            return []

        pairs = [(query, doc.content) for doc in documents]

        scores = self.model.predict(pairs)

        # Attach scores
        scored_docs = list(zip(documents, scores))

        # Sort descending by relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        return [doc for doc, _ in scored_docs[: self.top_k]]
