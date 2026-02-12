# src/retrieval/semantic_index_evaluator.py

from typing import List, Optional


class SemanticIndexEvaluator:
    """
    Evaluates retrieval quality for each clause.

    Metrics:
    - Coverage
    - Anchor Match
    - Noise Ratio
    - Groundedness
    """

    def evaluate(
        self,
        clause_result,
        evidence_docs: List[dict],
    ) -> dict:
        """
        Returns diagnostic signals used by aggregation and UI.
        """

        expected_sections = []

        if clause_result.statutory_basis:
            expected_sections = clause_result.statutory_basis.get("sections", [])

        # ------------------------------------
        # 1️⃣ Coverage
        # ------------------------------------
        coverage = len(evidence_docs) > 0

        # ------------------------------------
        # 2️⃣ Anchor Match
        # ------------------------------------
        anchor_match = False

        if expected_sections and evidence_docs:
            for doc in evidence_docs:
                text = doc.get("text", "").lower()
                for section in expected_sections:
                    if section.lower() in text:
                        anchor_match = True
                        break
                if anchor_match:
                    break

        # ------------------------------------
        # 3️⃣ Noise Ratio
        # ------------------------------------
        if evidence_docs:
            matching_docs = 0
            for doc in evidence_docs:
                text = doc.get("text", "").lower()
                if any(sec.lower() in text for sec in expected_sections):
                    matching_docs += 1

            noise_ratio = 1 - (matching_docs / len(evidence_docs))
        else:
            noise_ratio = 1.0

        # ------------------------------------
        # 4️⃣ Groundedness Score
        # ------------------------------------
        groundedness = (
            1.0 if anchor_match
            else 0.6 if coverage
            else 0.0
        )

        return {
            "coverage": coverage,
            "anchor_match": anchor_match,
            "noise_ratio": round(noise_ratio, 2),
            "groundedness": groundedness
        }