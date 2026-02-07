class ReasoningQualityScorer:

    def score(
        self,
        clause_result,
        explanation_text: str,
        evidence_pack
    ) -> float:

        score = 0.0

        # 1️⃣ Statutory grounding
        if any(
            kw in explanation_text.lower()
            for kw in ["rera", "act", "section", "authority"]
        ):
            score += 0.35

        # 2️⃣ Evidence support
        if evidence_pack.evidences:
            score += 0.25

        # 3️⃣ Causal clarity
        if any(
            kw in explanation_text.lower()
            for kw in ["because", "therefore", "results in", "leads to"]
        ):
            score += 0.25

        # 4️⃣ Language discipline
        if not any(
            kw in explanation_text.lower()
            for kw in ["illegal", "void", "unenforceable"]
        ):
            score += 0.15

        return round(score, 2)
