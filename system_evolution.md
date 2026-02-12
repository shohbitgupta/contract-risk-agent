# Contract Risk Analyzer — System Evolution & Component Rationale

This document explains **how the Contract Risk Analyzer evolved**, and **why each component exists** in the system. It is written for a technical audience (engineering + applied-ML + legal-tech).

---

## 1) What this system does (in one paragraph)

The Contract Risk Analyzer ingests a contract (PDF), splits it into legally meaningful chunks, interprets each clause into an **intent + statutory anchors**, retrieves **section-level statute/rule evidence** from FAISS vector indexes (optionally hybridized with lexical BM25 and reranked), evaluates retrieval quality/groundedness, generates a lawyer-grade clause explanation with citations, and finally aggregates clause outcomes into a contract-level score and a lawyer-friendly summary.

---

## 2) How we started (initial failure mode)

The system had a solid architecture (chunk → interpret → retrieve → explain → aggregate) but produced outputs dominated by **`insufficient_evidence`**. This strongly indicates the failure is usually **not reranking math**, but rather:

- **Index content**: statutes/rules were not chunked at “one section = one document”
- **Metadata quality**: missing/weak identifiers prevented deterministic citations and anchor matching
- **Anchor matching**: expected anchors like `Section 19(4)` did not match retrieved `Section 19` chunks
- **Evidence presentation**: summaries often cited template/model text (e.g. `model_bba`) even when a statute anchor existed, making the output feel ungrounded

---

## 3) Evolution milestones (what changed and why)

### A. Make the pipeline runnable (blocking errors)

- Fixed syntax/import/API mismatches around the reranker and orchestrator so the system can execute end-to-end.

**Why needed**: ground truth improvements require a working runtime loop.

### B. Statute indexing redesign (section-level chunks)

- Implemented statute parsing/indexing so that **one indexed document maps to one statutory section** (or one rule).
- Ensured each indexed document has required metadata (`source`, `chunk_id`) and statute fields (`act`, `section`, `section_number`, `state`, etc.).

**Why needed**: retrieval can only be as good as the “atomic units” in the vector store. Legal anchoring fails when a chunk contains multiple sections or is missing section identifiers.

### C. Rebuild + validate indexes (no guessing)

- Created repeatable rebuild scripts and a validation script that checks:
  - metadata completeness
  - “Section N / Rule N” retrieval spot checks

**Why needed**: legal RAG systems need **index QA** just like code QA.

### D. Hybrid retrieval (dense + lexical)

- Added lightweight BM25 pre-selection over dense-retrieved candidates, before cross-encoder reranking.

**Why needed**: statutes and rules often require lexical precision (e.g., “refund”, “interest”, “Rule 6”), which dense-only retrieval can miss or blur.

### E. Anchor correctness (rules + base-section matching)

- Treated state rules as first-class anchors (not only act sections).
- Made anchor matching robust to sub-sections: `Section 19(4)` matches the `Section 19` chunk (base-number match).

**Why needed**: intent rules and statutory basis often specify sub-sections while the index is section-level.

### F. Force anchor presence in final evidence set

- Ensured expected `Section N` / `Rule N` evidence is not dropped by reranking and top-k truncation.

**Why needed**: even perfect candidate recall is useless if the final evidence pack excludes the anchor, causing `anchor_match=False` and downstream “insufficient evidence”.

### G. Summary grounding correctness

- Ensured contract-level outputs (critical clauses, “high-risk clause count”) reflect the actual clause results rather than empty/incorrect issue lists.
- Improved “Evidence:” field preference to show statutory evidence when available (presentation-only).
- Added semantic consistency safeguards so verdict/ambiguity and score do not contradict each other.

**Why needed**: lawyer-facing outputs must be internally coherent and audit-friendly.

---

## 4) Component map (role + why it is needed)

Below is the canonical component list and rationale as implemented in this repository.

### 4.1 Ingestion and source preparation

#### `src/scripts/prepare_rera_sources.py`
- **Role**: downloads legal PDFs, extracts text (PDFMiner), falls back to OCR when needed, and normalizes structure.
- **Why needed**: statutes/rules frequently come from scanned PDFs; extraction quality is a prerequisite for correct section parsing.

#### `src/ingestion/statute_section_indexer.py`
- **Role**: parses full act/rule text into atomic documents and writes a FAISS index with strict metadata.
- **Why needed**: legal evidence must be **section-level** to support deterministic statutory anchoring and stable citation.

#### `src/scripts/rebuild_statute_index.py`
- **Role**: rebuilds statute indexes into the runtime directory structure.
- **Why needed**: enables repeatable, clean index regeneration after parsing/metadata changes.

#### `src/scripts/validate_statute_index.py`
- **Role**: QA for the statute index (metadata checks + sample queries).
- **Why needed**: prevents “silent failures” where retrieval works but returns unusable, unanchored chunks.

---

### 4.2 Vector indexing layer

#### `src/vector_index/index_base.py` (`IndexDocument`)
- **Role**: defines the atomic retrieval unit: `content` + `metadata` with required keys.
- **Why needed**: enforces minimum metadata guarantees (source, chunk_id) so downstream citation/normalization is possible.

#### `src/vector_index/faiss_index.py` (`FAISSVectorIndex`)
- **Role**: persists embeddings to `.faiss` and documents to `.meta.json`, and provides runtime `search()`.
- **Why needed**: deterministic local vector store with auditable sidecar metadata.

#### `src/vector_index/index_registry.py` (`IndexRegistry`)
- **Role**: loads and caches state-specific indexes from `src/data/vector_indexes/<state>`.
- **Why needed**: legal applicability is jurisdiction-dependent; indexes must be state-scoped and validated at runtime.

#### `src/vector_index/embedding.py` (`EmbeddingGenerator`)
- **Role**: stable embeddings using `SentenceTransformer(..., normalize_embeddings=True)`.
- **Why needed**: consistent embedding behavior is required for reproducible retrieval quality.

---

### 4.3 Contract chunking and preprocessing

#### `src/RAG/user_contract_chunker.py` (`UserContractChunker`)
- **Role**: splits contract text into `ContractChunk` objects with IDs, titles, and structure-aware grouping.
- **Why needed**: clause boundaries are critical. If chunks are too large or mis-split, intent detection and evidence grounding degrade.

#### `src/utils/chunk_filter.py`
- **Role**: filters out obvious non-semantic chunks (pure numbering, headers, tiny tokens).
- **Why needed**: reduces wasted LLM/retrieval calls and avoids polluting results with non-operative text.

---

### 4.4 Clause interpretation (intent + statutory basis)

#### `src/agents/clause_understanding_agent.py`
- **Role**: converts a clause chunk into a structured `ClauseUnderstandingResult` (intent, role, risk, statutory_basis).
- **Why needed**: retrieval must be guided by **legal intent** and **expected anchors** (sections/rules), not just raw similarity search.

#### `src/agents/intent_rules_engine.py`
- **Role**: policy/rules engine mapping intents to statutory anchors and retrieval guidance.
- **Why needed**: avoids “LLM guessing”; encodes jurisdiction-aware legal grounding expectations.

---

### 4.5 Retrieval (RAG) orchestration

#### `src/retrieval/retrieval_orchestrator.py` (`RetrievalOrchestrator`)
- **Role**:
  - build query text (intent + anchors + clause snippet)
  - vector search across prioritized indexes
  - optional BM25 hybrid preselection
  - cross-encoder rerank
  - enforce anchor presence in final evidence set
  - emit `EvidencePack` with diagnostics
- **Why needed**: legal RAG needs *both* recall (get the right statute section) and precision (rank the right evidence highest), plus traceability.

#### `src/retrieval/reranking_agent.py` (`CrossEncoderReRankingAgent`)
- **Role**: reranks candidate documents by query-document relevance using a cross-encoder.
- **Why needed**: improves semantic precision beyond embedding similarity, especially for dense statute text.

#### `src/retrieval/metadata_normalizer.py`
- **Role**: normalizes raw index metadata into the strict `ChunkMetadata` schema used across the system.
- **Why needed**: retrieval runs across heterogeneous corpora (act, rules, model agreements); normalization enables consistent UI + explanation formatting.

---

### 4.6 Retrieval quality evaluation (groundedness)

#### `src/utils/semantic_index_evaluator.py`
- **Role**: computes coverage/anchor match/noise penalty and emits a groundedness score used in explanations.
- **Why needed**: prevents the system from presenting confident legal conclusions when evidence is missing or anchors are not matched.

---

### 4.7 Clause explanation generation

#### `src/agents/legal_explanation_agent.py`
- **Role**: generates lawyer-grade clause explanations and alignment decisions using evidence + retrieval_quality.
- **Why needed**: transforms retrieval into a legally meaningful narrative with citations and conservative downgrade rules.

---

### 4.8 Aggregation and contract-level summary

#### `src/agents/contract_aggregation_agent.py` (`ContractAggregationAgent`)
- **Role**:
  - computes contract-level `overall_score` and `legal_confidence`
  - builds `top_issues` for lawyer summary and UI
- **Why needed**: contracts are evaluated holistically; aggregation converts clause-by-clause diagnostics into decision-grade contract signals.

#### `src/RAG/presentation/lawyer_summary_builder.py`
- **Role**: produces the lawyer-friendly summary (verdict, headline, reasons, critical clauses, next steps).
- **Why needed**: synthesizes technical outputs into a legally readable “opinion style” summary; enforces coherence between verdict and evidence clarity.

---

### 4.9 UI and entrypoint

#### `streamlit_app.py`
- **Role**: UI workflow, displays metrics and issues, highlights citations, and allows JSON download.
- **Why needed**: makes grounding visible (citations) and provides a reproducible artifact (JSON report).

#### `src/main.py`
- **Role**: orchestration entrypoint connecting extraction → chunking → interpret → retrieve → explain → aggregate → present.
- **Why needed**: single deterministic pipeline for CLI and Streamlit.

---

## 5) End-to-end data flow diagram (technical)

```mermaid
flowchart TD
  A[Streamlit UI: upload PDF] --> B[src/main.py: extract contract text]
  B --> C[UserContractChunker: ContractChunk list]
  C --> D[ClauseUnderstandingAgent + IntentRulesEngine: ClauseUnderstandingResult]
  D --> E[RetrievalOrchestrator]
  E --> E1[IndexRegistry: load state indexes]
  E1 --> E2[FAISS vector search]
  E2 --> E3[BM25 hybrid preselect (optional)]
  E3 --> E4[Cross-encoder rerank]
  E4 --> E5[Anchor injection + enforce anchors]
  E5 --> F[EvidencePack + diagnostics]
  F --> G[SemanticIndexEvaluator (utils): retrieval_quality]
  G --> H[LegalExplanationAgent: ClauseAnalysisResult]
  H --> I[ContractAggregationAgent: ContractAnalysisResult + top_issues]
  I --> J[Lawyer summary builder]
  J --> K[UI: metrics + issues + highlighted citations + JSON download]
```

---

## 6) Practical definition of “grounded” in this system

A clause is considered grounded when:

- an expected **Section/Rule anchor** is present in the final evidence pack (not just candidate pool),
- the evidence metadata provides a deterministic reference (`section`/`rule`/`chunk_id`),
- retrieval quality does not indicate high noise, and
- the explanation agent does not downgrade alignment due to missing coverage/anchor match.

---

## 7) Notes / known follow-ups

- **UP Rules “Rule 6” duplication**: the rules index contains multiple “Rule 6” entries. For maximum legal accuracy, store disambiguating metadata (e.g., chapter context) and adjust rule parsing to avoid collisions.
- Consider exposing a per-clause audit panel in UI showing:
  - expected anchors
  - matched anchors
  - final evidence list (statute first)

