# Contract Risk Analyzer - Technical Overview
## Complete System Architecture & Class Flow

---

## 📁 Project Structure

```
contract-risk-agent/
├── src/
│   ├── agents/                          # Core AI Agents
│   │   ├── clause_understanding_agent.py
│   │   ├── intent_rules_engine.py
│   │   ├── legal_explanation_agent.py
│   │   ├── legal_details_drafter_agent.py
│   │   ├── legal_details_verifier_agent.py
│   │   └── llm_analyzer_facade.py
│   │
│   ├── RAG/                             # Retrieval-Augmented Generation
│   │   ├── user_contract_chunker.py
│   │   ├── legal_data_chunker.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   └── metadata_emitter.py
│   │
│   ├── ingestion/                       # Document Ingestion Pipeline
│   │   ├── contract_parser/
│   │   │   ├── pdf_text_extractor.py
│   │   │   └── contract_ingestion.py
│   │   ├── ingestion_pipeline.py
│   │   └── rera_index_builder.py
│   │
│   ├── retrieval/                       # Vector Search & Retrieval
│   │   └── retrieval_orchestrator.py
│   │
│   ├── vector_index/                    # FAISS Vector Index Management
│   │   ├── index_registry.py
│   │   ├── faiss_index.py
│   │   ├── index_base.py
│   │   ├── index_writer.py
│   │   └── embedding.py
│   │
│   ├── tools/                           # Utility Functions
│   │   ├── logger.py
│   │   ├── llm_response_cache.py
│   │   ├── pdf_crawler.py
│   │   └── checksum.py
│   │
│   ├── audit/                           # Audit & Logging
│   │   └── audit_logger.py
│   │
│   ├── mcp_server/                      # MCP Server (External API)
│   │   └── mcp_server.py
│   │
│   ├── client/                          # MCP Client
│   │   └── mcp_client.py
│   │
│   ├── configs/                         # Configuration Files
│   │   ├── real_state_intent_rules.yaml
│   │   └── jurisdiction_requirements.yaml
│   │
│   ├── main.py                          # Entry Point
│   └── run_mcp.py                       # MCP Entry Point
│
├── data/
│   ├── sources/                         # Source Legal Documents
│   ├── vector_indexes/                  # FAISS Index Files
│   └── llm_cache/                       # LLM Response Cache
│
├── tests/                               # Test Suite
│   ├── test_clause_chunker.py
│   └── test_retrieval_coverage.py
│
├── eval/                                # Evaluation Framework
│   ├── gold_clauses.jsonl
│   ├── run_eval.py
│   └── thresholds.json
│
└── scripts/                             # Utility Scripts
    └── test_real_bba.py
```

---

## 🔄 System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT LAYER                              │
│  (PDF URL / Contract Text)                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              INGESTION LAYER                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ UserContractPDFExtractor                                 │   │
│  │  • extract_from_url()                                    │   │
│  │  • extract_from_file()                                   │   │
│  │  • _normalize()                                          │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ UserContractChunker                                       │   │
│  │  • chunk() - Splits contract into clauses                │   │
│  │  • _split_into_raw_clauses()                             │   │
│  │  • _detect_chunk_type()                                  │   │
│  │  Output: List[ContractChunk]                             │   │
│  └────────────────────┬───────────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            CLAUSE UNDERSTANDING LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ClauseUnderstandingAgent                                  │   │
│  │  • analyze(clause, state) → ClauseUnderstandingResult    │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ IntentRuleEngine                                          │   │
│  │  • analyze() - Matches intent from YAML rules            │   │
│  │  • _match_intent() - Keyword-based matching              │   │
│  │  • _evaluate_risk_level() - Risk assessment              │   │
│  │  • _build_retrieval_queries() - Query construction       │   │
│  │  • _infer_obligation_type() - Obligation classification │   │
│  └────────────────────┬───────────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RETRIEVAL LAYER (RAG)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ RetrievalOrchestrator                                     │   │
│  │  • retrieve(clause_result, state) → EvidencePack        │   │
│  │  • _resolve_indexes() - Selects vector indexes           │   │
│  │  • _passes_filters() - Metadata filtering                │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ IndexRegistry                                             │   │
│  │  • get_indexes(state) - Loads FAISS indexes              │   │
│  │  • validate_state() - Checks index availability          │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FAISSVectorIndex                                          │   │
│  │  • search(query_embedding, top_k) → List[IndexDocument] │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ EmbeddingGenerator                                        │   │
│  │  • embed(texts) → List[List[float]]                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            EXPLANATION GENERATION LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LegalExplanationAgent                                     │   │
│  │  • explain(clause, clause_result, evidence_pack)         │   │
│  │    → ExplanationResult                                   │   │
│  │  • _build_evidance() - Formats evidence for LLM          │   │
│  │  • _parse_and_validate_output() - Validates JSON         │   │
│  │  • _score_explanation() - Quality scoring                │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LegalLLMFacade                                            │   │
│  │  • explain() - Two-stage LLM pipeline                    │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│        ┌───────────────┴───────────────┐                        │
│        │                               │                        │
│        ▼                               ▼                        │
│  ┌──────────────┐            ┌──────────────────┐              │
│  │LocalLLMAdapter│            │ OpenAIRefiner    │              │
│  │• generate_draft()│          │• refine()        │              │
│  │  (Ollama)      │            │  (Ollama)       │              │
│  └──────────────┘            └──────────────────┘              │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│  ExplanationResult:                                              │
│    • clause_id, alignment, risk_level                            │
│    • summary, detailed_explanation                               │
│    • citations, quality_score, disclaimer                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Class Dependency Graph

```
ContractRiskAnalysisSystem (main.py)
    │
    ├──► UserContractChunker (RAG/user_contract_chunker.py)
    │       └──► ContractChunk (models)
    │
    ├──► ClauseUnderstandingAgent (agents/)
    │       └──► IntentRuleEngine
    │               └──► ClauseUnderstandingResult (models)
    │
    ├──► RetrievalOrchestrator (retrieval/)
    │       ├──► IndexRegistry (vector_index/)
    │       │       └──► FAISSVectorIndex
    │       │               └──► IndexDocument
    │       ├──► EmbeddingGenerator (vector_index/)
    │       └──► EvidencePack (models)
    │
    └──► LegalExplanationAgent (agents/)
            ├──► LegalLLMFacade
            │       ├──► LocalLLMAdapter
            │       └──► OpenAIRefiner
            ├──► LLMResponseCache (tools/)
            └──► ExplanationResult (models)
```

---

## 🏗️ Core Classes & Responsibilities

### 1. **ContractRiskAnalysisSystem** (`main.py`)
**Purpose**: Main orchestrator for end-to-end contract analysis

**Key Methods**:
- `__init__(index_registry, intent_rules_path)` - Initializes all agents
- `analyze_contract(contract_text, state)` - Runs full pipeline

**Dependencies**:
- UserContractChunker
- ClauseUnderstandingAgent
- RetrievalOrchestrator
- LegalExplanationAgent

---

### 2. **UserContractChunker** (`RAG/user_contract_chunker.py`)
**Purpose**: Splits contract text into structured clause chunks

**Key Methods**:
- `chunk(text: str) → List[ContractChunk]` - Main chunking method
- `_split_into_raw_clauses()` - Pattern-based clause detection
- `_detect_chunk_type()` - Classifies chunk type (CLAUSE, SCHEDULE, etc.)
- `_sub_chunk_definitions()` - Handles large definition blocks
- `_sub_chunk_schedule()` - Handles large schedules

**Output**: `ContractChunk` objects with:
- `chunk_id`, `text`, `chunk_type`, `title`, `confidence`

---

### 3. **ClauseUnderstandingAgent** (`agents/clause_understanding_agent.py`)
**Purpose**: Determines legal intent and risk level for each clause

**Key Methods**:
- `analyze(clause: ContractChunk, state: str) → ClauseUnderstandingResult`

**Delegates to**: `IntentRuleEngine`

---

### 4. **IntentRuleEngine** (`agents/intent_rules_engine.py`)
**Purpose**: YAML-driven intent matching and risk evaluation

**Key Methods**:
- `analyze(clause_id, clause_text) → ClauseUnderstandingResult`
- `_match_intent(clause_text)` - Keyword-based intent matching
- `_evaluate_risk_level(intent_name, clause_text)` - Risk assessment
  - HIGH: penalty, forfeit, termination, liability keywords
  - MEDIUM: financial obligations, dates, interest
  - LOW: standard terms, definitions
- `_build_retrieval_queries()` - Constructs vector search queries
- `_infer_obligation_type()` - Classifies obligation (PROMOTER/BUYER/SHARED)

**Configuration**: `configs/real_state_intent_rules.yaml`

---

### 5. **RetrievalOrchestrator** (`retrieval/retrieval_orchestrator.py`)
**Purpose**: Retrieves relevant legal evidence from vector indexes

**Key Methods**:
- `retrieve(clause_result, state) → EvidencePack`
- `_resolve_indexes(query, indexes)` - Selects which indexes to query
- `_passes_filters(metadata, filters, state)` - Applies metadata filters

**Process**:
1. Embeds retrieval query using `EmbeddingGenerator`
2. Searches each relevant index (rera_act, rera_rules, model_bba)
3. Filters by state and doc_type
4. Returns top-k evidence chunks

---

### 6. **IndexRegistry** (`vector_index/index_registry.py`)
**Purpose**: Manages state-specific FAISS vector indexes

**Key Methods**:
- `validate_state(state)` - Checks index availability
- `get_indexes(state) → Dict[str, FAISSVectorIndex]` - Loads indexes
- `list_states()` - Lists available states

**Index Types**:
- `rera_act` - RERA Act 2016 sections
- `rera_rules` - State-specific RERA rules
- `model_bba` - Model Builder-Buyer Agreements

---

### 7. **FAISSVectorIndex** (`vector_index/faiss_index.py`)
**Purpose**: FAISS-based vector similarity search

**Key Methods**:
- `search(query_embedding, top_k) → List[IndexDocument]`
- `add(embeddings, documents)` - Index building
- `persist()` - Saves to disk
- `load(index_path, dim)` - Loads from disk

**Storage**:
- `.faiss` - FAISS index file
- `.meta.json` - Metadata sidecar

---

### 8. **LegalExplanationAgent** (`agents/legal_explanation_agent.py`)
**Purpose**: Generates human-readable explanations from evidence

**Key Methods**:
- `explain(clause, clause_result, evidence_pack) → ExplanationResult`
- `_build_evidance()` - Formats evidence for LLM prompt
- `_parse_and_validate_output()` - Validates JSON output
- `_score_explanation()` - Calculates quality score (0.0-1.0)
- `_determine_alignment()` - Sets alignment status

**Quality Scoring**:
- Evidence coverage (30%)
- Authority strength (30% for RERA Act, 20% for rules)
- Jurisdiction correctness (20%)
- Uncertainty honesty (20%)

---

### 9. **LegalLLMFacade** (`agents/llm_analyzer_facade.py`)
**Purpose**: Two-stage LLM pipeline (draft + refine)

**Key Methods**:
- `explain(clause_text, evidence_text) → str`

**Pipeline**:
1. **LocalLLMAdapter** - Generates draft using Ollama (llama3:8b)
2. **OpenAIRefiner** - Refines draft into strict JSON using Ollama

---

### 10. **OpenAIRefiner** (`agents/legal_details_verifier_agent.py`)
**Purpose**: Validates and normalizes LLM output using Pydantic

**Key Methods**:
- `refine(draft_text, clause_text, evidence_text) → str`
- `_extract_json(text)` - Extracts JSON from raw output
- `_normalize_output(parsed)` - Pydantic validation

**Pydantic Models**:
- `LLMOutput` - Validates alignment, key_findings, explanation, evidence_mapping
- `EvidenceMapping` - Validates claim-to-evidence mapping

**Fallback**: Returns safe defaults if validation fails

---

### 11. **UserContractPDFExtractor** (`ingestion/contract_parser/pdf_text_extractor.py`)
**Purpose**: Extracts text from PDF contracts

**Key Methods**:
- `extract_from_url(pdf_url) → str`
- `extract_from_file(pdf_path) → str`
- `_normalize(text)` - Cleans extracted text

**Libraries**: `pdfminer`

---

### 12. **LLMResponseCache** (`tools/llm_response_cache.py`)
**Purpose**: Caches LLM responses to avoid redundant API calls

**Key Methods**:
- `get(cache_key) → Optional[Dict]`
- `set(cache_key, value)`
- `build_cache_key(clause_text, intent, obligation_type, evidence_pack)`

**Cache Key**: SHA256 hash of clause + intent + evidence fingerprint

---

## 🔄 Data Flow Example

### Example: Analyzing a "Delay in Possession" Clause

```
1. INPUT: "The Promoter shall hand over possession by the agreed date..."

2. CHUNKING (UserContractChunker):
   → ContractChunk(
       chunk_id="7.1",
       text="The Promoter shall hand over possession...",
       chunk_type=ChunkType.CLAUSE
     )

3. INTENT MATCHING (IntentRuleEngine):
   → ClauseUnderstandingResult(
       intent="possession_delay",
       risk_level="high",  # Contains "delay" keyword
       obligation_type="PROMOTER_OBLIGATION",
       retrieval_queries=[{
         "index": "rera_act",
         "intent": "delay in possession",
         "filters": {"doc_type": ["rera_act"]}
       }]
     )

4. RETRIEVAL (RetrievalOrchestrator):
   → EvidencePack(
       evidences=[
         Evidence(
           source="RERA Act 2016",
           section_or_clause="Section 18",
           text="If the promoter fails to complete...",
           metadata={"doc_type": "rera_act", "state": "uttar_pradesh"}
         )
       ]
     )

5. EXPLANATION (LegalExplanationAgent):
   → ExplanationResult(
       clause_id="7.1",
       alignment="aligned",
       risk_level="high",
       summary="This clause aligns with RERA Section 18...",
       detailed_explanation="The contract clause requires...",
       citations=[{"source": "RERA Act 2016", "section_or_clause": "Section 18"}],
       quality_score=0.85
     )
```

---

## 🎯 Key Design Patterns

### 1. **Pipeline Pattern**
Each stage processes input and passes structured output to the next stage.

### 2. **Strategy Pattern**
`IntentRuleEngine` uses YAML-based rules for intent matching (easily extensible).

### 3. **Facade Pattern**
`LegalLLMFacade` simplifies the two-stage LLM pipeline.

### 4. **Repository Pattern**
`IndexRegistry` abstracts vector index access.

### 5. **Caching Pattern**
`LLMResponseCache` prevents redundant LLM calls.

---

## 📦 External Dependencies

### Core Libraries:
- **pydantic** - Data validation
- **faiss-cpu** - Vector similarity search
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **pdfminer** - PDF text extraction
- **yaml** - Configuration parsing
- **mcp** - Model Context Protocol server

### LLM:
- **Ollama** (local) - llama3:8b for draft generation and refinement

---

## 🔐 Configuration Files

### `configs/real_state_intent_rules.yaml`
Defines:
- Intent keywords (possession_delay, refund_and_withdrawal, etc.)
- Retrieval index mappings
- Document type filters

### `configs/jurisdiction_requirements.yaml`
Defines:
- Mandatory intents per jurisdiction
- State-specific requirements

---

## 🧪 Testing & Evaluation

### Test Files:
- `tests/test_clause_chunker.py` - Chunking accuracy
- `tests/test_retrieval_coverage.py` - Retrieval quality

### Evaluation:
- `eval/gold_clauses.jsonl` - 30 labeled test clauses
- `eval/run_eval.py` - Automated evaluation runner
- `eval/thresholds.json` - Quality gates

---

## 🚀 Entry Points

### 1. **CLI** (`main.py`)
```bash
python src/main.py
```

### 2. **MCP Server** (`run_mcp.py`)
```bash
python src/run_mcp.py --mode server
```

### 3. **MCP Client** (`run_mcp.py`)
```bash
python src/run_mcp.py --mode pdf --pdf-url <URL> --state uttar_pradesh
```

### 4. **Test Script** (`scripts/test_real_bba.py`)
```bash
python src/scripts/test_real_bba.py --pdf-url <URL> --state uttar_pradesh
```

---

## 📈 Performance Characteristics

### Latency (per clause):
- PDF Extraction: ~2-5s
- Chunking: <100ms
- Intent Matching: <50ms
- Retrieval: ~200-500ms (depends on index size)
- LLM Explanation: ~5-15s (Ollama local)
- **Total per clause**: ~8-20s

### Caching:
- LLM responses cached by clause+intent+evidence fingerprint
- Cache hit rate: ~60-80% for similar clauses

### Scalability:
- Vector indexes: O(n) search time, supports millions of chunks
- LLM: Local Ollama, no API rate limits
- State-specific indexes: Loaded on-demand, cached in memory

---

## 🔍 Monitoring Points

### Key Metrics:
1. **Chunking Quality**: Clause boundary accuracy
2. **Intent Accuracy**: % correct intent matches
3. **Retrieval Precision**: % relevant evidence retrieved
4. **LLM Quality Score**: Average explanation quality (0.0-1.0)
5. **Citation Coverage**: % clauses with valid citations
6. **Risk Distribution**: High/Medium/Low risk breakdown

---

## 🛠️ Extension Points

### Adding New Intent:
1. Add intent definition to `configs/real_state_intent_rules.yaml`
2. Add keywords and retrieval config
3. System automatically picks it up

### Adding New State:
1. Add state to `ingestion/config/registry.json`
2. Run `ingestion_pipeline.py` to download documents
3. Run `rera_index_builder.py` to build indexes
4. System automatically supports it

### Custom LLM:
1. Modify `LocalLLMAdapter` or `OpenAIRefiner`
2. Update model name/API endpoint
3. Ensure JSON output format matches

---

## 📝 Notes for Tech Team

### Critical Path:
1. **Chunking accuracy** directly impacts downstream analysis
2. **Intent matching** determines which legal sources are searched
3. **Retrieval quality** determines explanation accuracy
4. **LLM output validation** prevents hallucination

### Known Limitations:
- Chunking relies on regex patterns (may miss non-standard formats)
- Intent matching is keyword-based (may misclassify complex clauses)
- Risk level is heuristic-based (not legal advice)
- LLM may occasionally produce invalid JSON (handled with fallbacks)

### Future Improvements:
- ML-based clause segmentation
- Fine-tuned intent classifier
- Multi-hop retrieval for complex queries
- Structured output LLMs (JSON mode)

---

**Last Updated**: 2026-02-01
**Version**: 1.0
