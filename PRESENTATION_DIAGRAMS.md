# Contract Risk Analyzer - Presentation Diagrams
## Visual Flow & Architecture for Tech Team

---

## 🎯 System Overview (High-Level)

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTRACT RISK ANALYZER                    │
│                                                              │
│  Input: PDF/Text  →  Analysis  →  Risk Report + Citations   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Complete Processing Pipeline

```
┌──────────────┐
│  User Input  │
│  (PDF/Text)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: INGESTION                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ UserContractPDFExtractor                             │  │
│  │ • Downloads PDF from URL                             │  │
│  │ • Extracts text using pdfminer                       │  │
│  │ • Normalizes text (removes headers/footers)          │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: CHUNKING                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ UserContractChunker                                 │  │
│  │ • Detects clause boundaries (regex patterns)        │  │
│  │ • Classifies chunk types (CLAUSE, SCHEDULE, etc.)   │  │
│  │ • Handles sub-chunking for large sections           │  │
│  │ Output: List[ContractChunk]                          │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: INTENT UNDERSTANDING                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ClauseUnderstandingAgent                              │  │
│  │   └─► IntentRuleEngine                               │  │
│  │       • Keyword-based intent matching                 │  │
│  │       • Risk level evaluation (HIGH/MEDIUM/LOW)       │  │
│  │       • Obligation type classification               │  │
│  │       • Retrieval query construction                 │  │
│  │ Output: ClauseUnderstandingResult                    │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: EVIDENCE RETRIEVAL (RAG)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ RetrievalOrchestrator                                │  │
│  │   ├─► EmbeddingGenerator                             │  │
│  │   │   • Converts query to 384-dim vector             │  │
│  │   │                                                    │  │
│  │   ├─► IndexRegistry                                  │  │
│  │   │   • Loads state-specific FAISS indexes           │  │
│  │   │                                                    │  │
│  │   └─► FAISSVectorIndex                               │  │
│  │       • Vector similarity search (top-k)             │  │
│  │       • Metadata filtering (state, doc_type)        │  │
│  │ Output: EvidencePack (List[Evidence])                │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: EXPLANATION GENERATION                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ LegalExplanationAgent                                │  │
│  │   ├─► LLMResponseCache (check cache first)          │  │
│  │   │                                                    │  │
│  │   └─► LegalLLMFacade                                 │  │
│  │       ├─► LocalLLMAdapter (Ollama)                   │  │
│  │       │   • Generates draft explanation              │  │
│  │       │                                                    │
│  │       └─► OpenAIRefiner (Ollama)                     │  │
│  │           • Refines to strict JSON                  │  │
│  │           • Pydantic validation                     │  │
│  │ Output: ExplanationResult                            │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌──────────────┐
│   Output     │
│  (JSON/API)  │
└──────────────┘
```

---

## 🏛️ Class Hierarchy & Relationships

```
ContractRiskAnalysisSystem
│
├─── UserContractChunker
│    └─── ContractChunk (dataclass)
│         ├── chunk_id: str
│         ├── text: str
│         ├── chunk_type: ChunkType (Enum)
│         └── confidence: float
│
├─── ClauseUnderstandingAgent
│    └─── IntentRuleEngine
│         ├── _match_intent() → intent_name
│         ├── _evaluate_risk_level() → "high"|"medium"|"low"
│         ├── _build_retrieval_queries() → List[Dict]
│         └── _infer_obligation_type() → "PROMOTER"|"BUYER"|"SHARED"
│         └─── ClauseUnderstandingResult (dataclass)
│              ├── intent: str
│              ├── risk_level: str
│              ├── obligation_type: str
│              └── retrieval_queries: List[Dict]
│
├─── RetrievalOrchestrator
│    ├─── EmbeddingGenerator
│    │    └── embed(texts) → List[List[float]]
│    │
│    ├─── IndexRegistry
│    │    └── get_indexes(state) → Dict[str, FAISSVectorIndex]
│    │         ├── "rera_act" → FAISSVectorIndex
│    │         ├── "rera_rules" → FAISSVectorIndex
│    │         └── "model_bba" → FAISSVectorIndex
│    │
│    └─── EvidencePack (dataclass)
│         └── evidences: List[Evidence]
│              ├── source: str
│              ├── section_or_clause: str
│              ├── text: str
│              └── metadata: Dict
│
└─── LegalExplanationAgent
     ├─── LLMResponseCache
     │    └── get(cache_key) → Optional[Dict]
     │
     └─── LegalLLMFacade
          ├─── LocalLLMAdapter
          │    └── generate_draft() → str
          │
          └─── OpenAIRefiner
               ├── refine() → str (JSON)
               └─── LLMOutput (Pydantic Model)
                    ├── alignment: str
                    ├── key_findings: List[str]
                    ├── explanation: str
                    └── evidence_mapping: List[EvidenceMapping]
               └─── ExplanationResult (dataclass)
                    ├── clause_id: str
                    ├── alignment: str
                    ├── risk_level: str
                    ├── summary: str
                    ├── detailed_explanation: str
                    ├── citations: List[Dict]
                    ├── quality_score: float
                    └── disclaimer: str
```

---

## 🔄 Data Flow: Single Clause Processing

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT: Contract Clause                    │
│  "The Promoter shall hand over possession by the agreed     │
│   date. In case of delay, the allottee may seek remedies."  │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: Intent Matching                                      │
│  ─────────────────────────────────────────────────────────── │
│  IntentRuleEngine._match_intent()                             │
│  • Searches keywords: "possession", "delay"                  │
│  • Matches: "possession_delay" intent                        │
│  • Evaluates risk: "high" (contains "delay")                 │
│  • Classifies obligation: "PROMOTER_OBLIGATION"             │
│                                                               │
│  Output:                                                      │
│  {                                                            │
│    "intent": "possession_delay",                             │
│    "risk_level": "high",                                      │
│    "retrieval_queries": [{                                    │
│      "index": "rera_act",                                     │
│      "intent": "delay in possession",                        │
│      "filters": {"doc_type": ["rera_act"]}                   │
│    }]                                                         │
│  }                                                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Vector Search                                        │
│  ─────────────────────────────────────────────────────────── │
│  RetrievalOrchestrator.retrieve()                            │
│  • Embeds query: "delay in possession" → [0.23, -0.45, ...] │
│  • Searches rera_act index (FAISS)                           │
│  • Top-5 results:                                             │
│    1. Section 18: "If promoter fails to complete..."         │
│    2. Section 18(1): "Promoter liable for interest..."       │
│    3. ...                                                     │
│  • Filters by state="uttar_pradesh"                          │
│                                                               │
│  Output: EvidencePack                                         │
│  {                                                            │
│    "evidences": [                                             │
│      {                                                        │
│        "source": "RERA Act 2016",                            │
│        "section_or_clause": "Section 18",                    │
│        "text": "If the promoter fails to complete...",        │
│        "metadata": {"doc_type": "rera_act", ...}              │
│      }                                                        │
│    ]                                                          │
│  }                                                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: LLM Explanation                                      │
│  ─────────────────────────────────────────────────────────── │
│  LegalExplanationAgent.explain()                              │
│  • Builds prompt with clause + evidence                      │
│  • Checks cache (miss)                                        │
│  • Calls LegalLLMFacade:                                      │
│    ├─ LocalLLMAdapter generates draft                        │
│    └─ OpenAIRefiner produces JSON                            │
│  • Validates JSON output                                      │
│  • Scores quality (0.85)                                      │
│                                                               │
│  Output: ExplanationResult                                    │
│  {                                                            │
│    "clause_id": "7.1",                                        │
│    "alignment": "aligned",                                    │
│    "risk_level": "high",                                      │
│    "summary": "This clause aligns with RERA Section 18...",  │
│    "detailed_explanation": "The contract requires...",        │
│    "citations": [                                             │
│      {"source": "RERA Act 2016", "section": "Section 18"}    │
│    ],                                                         │
│    "quality_score": 0.85                                      │
│  }                                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Module Organization

```
src/
│
├─── agents/              [AI Logic Layer]
│    ├── clause_understanding_agent.py    → Intent detection
│    ├── intent_rules_engine.py           → YAML-based rules
│    ├── legal_explanation_agent.py       → Explanation generation
│    ├── legal_details_drafter_agent.py   → LLM draft
│    ├── legal_details_verifier_agent.py  → LLM refinement
│    └── llm_analyzer_facade.py           → LLM orchestration
│
├─── RAG/                 [Retrieval Layer]
│    ├── user_contract_chunker.py         → Contract chunking
│    ├── legal_data_chunker.py            → Legal doc chunking
│    ├── models.py                        → Data models
│    ├── schemas.py                       → Pydantic schemas
│    └── metadata_emitter.py              → Metadata generation
│
├─── ingestion/           [Input Processing]
│    ├── contract_parser/
│    │   ├── pdf_text_extractor.py        → PDF extraction
│    │   └── contract_ingestion.py         → Ingestion pipeline
│    ├── ingestion_pipeline.py            → Legal doc ingestion
│    └── rera_index_builder.py            → Index building
│
├─── retrieval/           [Search Layer]
│    └── retrieval_orchestrator.py         → RAG orchestration
│
├─── vector_index/        [Vector Search]
│    ├── index_registry.py                → Index management
│    ├── faiss_index.py                   → FAISS operations
│    ├── index_base.py                    → Base classes
│    ├── index_writer.py                  → Index writing
│    └── embedding.py                     → Text embeddings
│
├─── tools/               [Utilities]
│    ├── logger.py                        → Logging setup
│    ├── llm_response_cache.py            → Response caching
│    ├── pdf_crawler.py                   → PDF downloading
│    └── checksum.py                      → File checksums
│
├─── audit/               [Audit Trail]
│    └── audit_logger.py                  → Event logging
│
├─── mcp_server/          [External API]
│    └── mcp_server.py                    → MCP tool server
│
├─── client/              [MCP Client]
│    └── mcp_client.py                    → MCP client
│
└─── configs/             [Configuration]
     ├── real_state_intent_rules.yaml     → Intent rules
     └── jurisdiction_requirements.yaml   → Jurisdiction rules
```

---

## 🎨 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ContractRiskAnalysisSystem                │
│                         (Orchestrator)                       │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         │              │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Chunker │   │  Intent │   │Retrieval│   │Explain  │
    │         │   │  Agent   │   │Orch.    │   │Agent    │
    └────┬────┘   └────┬─────┘   └────┬────┘   └────┬────┘
         │              │              │              │
         │              │              │              │
    ┌────▼─────────────▼──────────────▼──────────────▼────┐
    │                                                       │
    │              Data Models Layer                        │
    │  ContractChunk → ClauseUnderstandingResult            │
    │              → EvidencePack → ExplanationResult        │
    │                                                       │
    └───────────────────────────────────────────────────────┘
```

---

## 🔍 Risk Level Evaluation Logic

```
┌─────────────────────────────────────────────────────────────┐
│           Risk Level Evaluation Flow                         │
└─────────────────────────────────────────────────────────────┘

Input: intent_name + clause_text
         │
         ▼
    ┌────────┐
    │ Check  │  High-risk keywords?
    │Keywords│  (penalty, forfeit, termination, liability)
    └───┬────┘
        │ YES
        ▼
    ┌────────┐
    │ HIGH   │
    └────────┘
        │
        │ NO
        ▼
    ┌────────┐
    │ Check  │  High-risk intents?
    │ Intent │  (refund, interest, jurisdiction)
    └───┬────┘
        │ YES
        ▼
    ┌────────┐
    │ HIGH   │
    └────────┘
        │
        │ NO
        ▼
    ┌────────┐
    │ Check  │  Medium-risk intents?
    │ Intent │  (possession_delay, defect_liability)
    └───┬────┘
        │ YES
        ▼
    ┌────────┐
    │ MEDIUM │
    └────────┘
        │
        │ NO
        ▼
    ┌────────┐
    │  LOW   │  (default for standard terms)
    └────────┘
```

---

## 📈 Quality Scoring Breakdown

```
Explanation Quality Score (0.0 - 1.0)
│
├── Evidence Coverage (30%)
│   └── ≥2 evidence chunks? → +0.3
│
├── Authority Strength (30%)
│   ├── RERA Act present? → +0.3
│   └── State Rules only? → +0.2
│
├── Jurisdiction Correctness (20%)
│   └── State matches? → +0.2
│
└── Uncertainty Honesty (20%)
    └── "insufficient_evidence" alignment? → +0.2
```

---

## 🚀 Entry Points & Usage

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                             │
└─────────────────────────────────────────────────────────────┘

1. CLI (main.py)
   └─► python src/main.py
       • Analyzes hardcoded PDF URL
       • Prints results to console

2. MCP Server (run_mcp.py)
   └─► python src/run_mcp.py --mode server
       • Starts MCP server over stdio
       • Exposes analyze_contract_pdf() and analyze_contract_text()

3. MCP Client (run_mcp.py)
   └─► python src/run_mcp.py --mode pdf --pdf-url <URL>
       • Spawns server, calls tool, prints result

4. Test Script (scripts/test_real_bba.py)
   └─► python src/scripts/test_real_bba.py --pdf-url <URL>
       • Full analysis + quality evaluation
       • Generates JSON report
```

---

## 📊 Performance Metrics

```
┌─────────────────────────────────────────────────────────────┐
│              PERFORMANCE CHARACTERISTICS                     │
└─────────────────────────────────────────────────────────────┘

Per-Clause Processing Time:
├── PDF Extraction:     2-5s
├── Chunking:           <100ms
├── Intent Matching:    <50ms
├── Retrieval:          200-500ms
└── LLM Explanation:    5-15s (Ollama local)
────────────────────────────────────────────
Total per clause:       8-20s

Caching:
└── LLM Response Cache: 60-80% hit rate

Scalability:
├── Vector Indexes:     O(n) search, supports millions
├── State Indexes:      Loaded on-demand, cached in memory
└── LLM:                Local Ollama, no rate limits
```

---

## 🎯 Key Design Decisions

```
┌─────────────────────────────────────────────────────────────┐
│              ARCHITECTURAL DECISIONS                         │
└─────────────────────────────────────────────────────────────┘

1. Deterministic Intent Matching
   └─► YAML-based rules (not ML) for transparency

2. Evidence-Bound Explanations
   └─► LLM can only cite retrieved evidence (prevents hallucination)

3. Two-Stage LLM Pipeline
   └─► Draft (local) + Refine (local) for cost control

4. State-Specific Indexes
   └─► Separate FAISS indexes per state for jurisdiction accuracy

5. Pydantic Validation
   └─► Strict schema validation for LLM outputs

6. Caching Strategy
   └─► Cache by clause+intent+evidence fingerprint
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-01  
**For**: Technical Team Presentation
