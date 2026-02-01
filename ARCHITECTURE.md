# Contract Risk Agent Architecture

This document describes the system architecture, processing flow, and key
components of the AI contract risk analyzer.

## 1) End-to-End Flow

1. **Input**
   - User uploads a Builder–Buyer Agreement (BBA) PDF or provides text.

2. **Contract Text Extraction**
   - `UserContractPDFExtractor` downloads PDFs and extracts text.
   - Normalizes the text while preserving legal structure.

3. **Clause Chunking**
   - `UserContractChunker` splits the contract into clause-sized chunks.
   - Outputs `ContractChunk` objects with ids, text, and metadata.

4. **Clause Understanding**
   - `ClauseUnderstandingAgent` + `IntentRuleEngine`
   - Determines clause intent using YAML rules and emits retrieval queries.

5. **Retrieval (RAG Layer)**
   - `RetrievalOrchestrator` queries FAISS vector indexes.
   - Filters by jurisdiction/state and document type.
   - Returns an `EvidencePack` per clause.

6. **Legal Explanation**
   - `LegalExplanationAgent` uses evidence to produce an explanation.
   - Outputs `ExplanationResult` with alignment, risk, citations, and disclaimer.

7. **Output**
   - Structured response for UI/API.

## 2) Core Modules

### Ingestion
- `ingestion/contract_parser/pdf_text_extractor.py`
  - PDF download + text extraction with validation.
- `ingestion/contract_parser/contract_ingestion.py`
  - Wires PDF → normalized text → clause chunking.

### Clause Chunking
- `RAG/user_contract_chunker.py`
  - Regex-driven clause detection.
  - Handles schedules, definitions, notes, and long sections.

### Clause Understanding
- `agents/intent_rules_engine.py`
  - Loads `configs/real_state_intent_rules.yaml`.
  - Keyword-based intent matching.
- `agents/clause_understanding_agent.py`
  - Orchestrates intent detection and retrieval query emission.

### Retrieval
- `retrieval/retrieval_orchestrator.py`
  - Embeds clause intent and queries vector indexes.
  - Hard filters by state and document type.
- `vector_index/`
  - FAISS-based search with sidecar metadata.

### Legal Explanation
- `agents/legal_explanation_agent.py`
  - Evidence-bound response generation.
  - Enforces strict JSON output and citation mapping.
- `agents/legal_details_drafter_agent.py`
  - Local draft via Ollama (optional).
- `agents/legal_details_verifier_agent.py`
  - OpenAI refiner returns strict JSON.

## 3) Data & Indexes

### Sources
- `data/sources/uttar_pradesh/`
  - RERA Act, UP RERA Rules, Model BBA, circulars, case law.

### Vector Indexes
- `data/vector_indexes/uttar_pradesh/`
  - `rera_act.faiss`
  - `rera_rules.faiss`
  - `model_bba.faiss`

### Index Build
- `ingestion/rera_index_builder.py`
  - Chunks legal documents and builds FAISS indexes.

## 4) Configuration

- `configs/real_state_intent_rules.yaml`
  - Clause intent rules, retrieval targets, and filters.
- `configs/jurisdiction_requirements.yaml`
  - Required intent coverage per jurisdiction.

## 5) Key Interfaces

### Domain Models (`RAG/models.py`)
- `ContractChunk` (user contract)
- `ClauseUnderstandingResult`
- `Evidence`, `EvidencePack`
- `ExplanationResult`

## 6) Guardrails & Constraints

- **Evidence-bound explanations**: responses must reference retrieved evidence.
- **No legal advice**: disclaimer included in `ExplanationResult`.
- **Strict JSON output**: explanation agent validates outputs.
- **Jurisdiction filters**: retrieval filters by state and doc type.

## 7) Running the Pipeline

### CLI (main)
```
pipenv run python src/main.py
```

### MCP (if enabled)
```
pipenv run python src/run_mcp.py --mode server
```

## 8) Testing

```
pipenv run pytest
```

Notes:
- Retrieval tests are skipped if vector indexes or dependencies are missing.

## 9) Future Extensions

- OCR fallback for scanned PDFs
- Better clause boundary detection
- Explicit uncertainty reporting
- Expanded jurisdiction support
