# DocOps Agent - Architecture

## System Architecture Diagram

```mermaid
flowchart TB
    subgraph Input["Document Input"]
        PDF[PDF Files]
        DOCX[Word Documents]
        MD[Markdown Files]
        TXT[Text Files]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        Parser["Parser<br/>Extract Text & Structure"]
        Chunker["Chunker<br/>Section-Aware Splitting"]
        Embedder["Embedder<br/>all-MiniLM-L6-v2<br/>384 dimensions"]
    end

    subgraph Elasticsearch["Elasticsearch 8.12"]
        subgraph Indexes["Indexes"]
            DocsIndex["docops-documents<br/>━━━━━━━━━━━━━━<br/>• BM25 text search<br/>• Document metadata"]
            ChunksIndex["docops-chunks<br/>━━━━━━━━━━━━━━<br/>• dense_vector (384d)<br/>• Hybrid search"]
            AlertsIndex["docops-alerts<br/>━━━━━━━━━━━━━━<br/>• Severity routing<br/>• Deduplication"]
        end
        HybridSearch["Hybrid Search<br/>BM25 + kNN"]
        Aggregations["Aggregations<br/>Analytics & Stats"]
    end

    subgraph AgentLayer["Multi-Step Agent Layer"]
        AgentCore["Agent Core<br/>Intent Detection & Routing"]

        subgraph Tools["6 Agent Tools"]
            T1["search_documents"]
            T2["compare_sections"]
            T3["run_consistency_check"]
            T4["generate_report"]
            T5["create_alert"]
            T6["get_document_health"]
        end

        ReviewerAgent["Reviewer Agent<br/>Verify Findings"]

        subgraph Loop["Multi-Step Reasoning Loop"]
            Step1["1. Analyze Query"]
            Step2["2. Select Tools"]
            Step3["3. Execute Tools"]
            Step4["4. Verify Results"]
            Step5["5. Synthesize Response"]
        end
    end

    subgraph Analysis["Analysis Engine"]
        ConflictDetector["Conflict Detector<br/>Numeric, Policy, Date"]
        StalenessChecker["Staleness Checker<br/>Expired & Outdated"]
        GapAnalyzer["Gap Analyzer<br/>Coverage Gaps"]
    end

    subgraph Output["Output Layer"]
        Alerts["Alerts<br/>Critical/High/Medium/Low"]
        Reports["Reports<br/>Markdown, Excel, PDF"]
        SlackNotify["Slack Notifications<br/>Webhook Integration"]
    end

    subgraph Frontend["Frontend Layer"]
        StreamlitUI["Streamlit UI<br/>━━━━━━━━━━━━━━<br/>• Dashboard<br/>• Agent Chat<br/>• Conflict Viewer<br/>• Reports<br/>• Search<br/>• Upload"]
        FastAPI["FastAPI Backend<br/>REST Endpoints"]
    end

    subgraph User["User Interaction"]
        UserQuery["User Query"]
        UserDocs["Upload Documents"]
    end

    %% Connections - Input Flow
    PDF & DOCX & MD & TXT --> Parser
    Parser --> Chunker
    Chunker --> Embedder
    Embedder --> ChunksIndex
    Parser --> DocsIndex

    %% Elasticsearch Internal
    ChunksIndex --> HybridSearch
    DocsIndex --> HybridSearch
    ChunksIndex --> Aggregations
    AlertsIndex --> Aggregations

    %% Agent Flow
    UserQuery --> StreamlitUI
    StreamlitUI --> FastAPI
    FastAPI --> AgentCore
    AgentCore --> Step1
    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Tools
    Tools --> HybridSearch
    Tools --> Analysis
    Step3 --> Step4
    Step4 --> ReviewerAgent
    ReviewerAgent --> Step5
    Step5 --> AgentCore

    %% Analysis connections
    HybridSearch --> ConflictDetector
    HybridSearch --> StalenessChecker
    HybridSearch --> GapAnalyzer
    ConflictDetector & StalenessChecker & GapAnalyzer --> T3

    %% Output Flow
    T5 --> AlertsIndex
    T5 --> SlackNotify
    T4 --> Reports
    AlertsIndex --> Alerts

    %% User Document Upload
    UserDocs --> StreamlitUI
    StreamlitUI --> Parser

    %% Styling
    classDef elastic fill:#005571,stroke:#00bfb3,color:#fff
    classDef agent fill:#7b68ee,stroke:#483d8b,color:#fff
    classDef tool fill:#ff6b6b,stroke:#c0392b,color:#fff
    classDef output fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef frontend fill:#3498db,stroke:#2980b9,color:#fff

    class DocsIndex,ChunksIndex,AlertsIndex,HybridSearch,Aggregations elastic
    class AgentCore,ReviewerAgent,Step1,Step2,Step3,Step4,Step5 agent
    class T1,T2,T3,T4,T5,T6 tool
    class Alerts,Reports,SlackNotify output
    class StreamlitUI,FastAPI frontend
```

## Component Details

### 1. Document Ingestion Pipeline

| Component | Technology | Purpose |
|-----------|------------|---------|
| Parser | PyMuPDF, python-docx | Extract text from various formats |
| Chunker | Custom | Section-aware text splitting |
| Embedder | sentence-transformers | Generate 384-dim vectors |

### 2. Elasticsearch Storage

| Index | Content | Search Type |
|-------|---------|-------------|
| docops-documents | Full documents, metadata | BM25 text search |
| docops-chunks | Text chunks + embeddings | Hybrid (BM25 + kNN) |
| docops-alerts | Alerts with deduplication | Filtered queries |

### 3. Multi-Step Agent

The agent follows this reasoning loop for each query:

```
User Query
    ↓
[1] Analyze Intent (conflict? staleness? search?)
    ↓
[2] Select Tools (1-4 tools based on intent)
    ↓
[3] Execute Tools (search → analyze → alert)
    ↓
[4] Verify Results (Reviewer Agent confirms findings)
    ↓
[5] Synthesize Response (combine results + suggestions)
    ↓
Final Answer + Step Trace
```

### 4. Agent Tools

| Tool | Purpose | ES Operations |
|------|---------|---------------|
| search_documents | Find relevant content | Hybrid search |
| compare_sections | Compare two sections | Multi-get |
| run_consistency_check | Detect conflicts | Aggregations |
| generate_report | Create reports | Full scan |
| create_alert | Flag issues | Index + Slack |
| get_document_health | Corpus metrics | Aggregations |

### 5. Analysis Engine

- **Conflict Detector**: Finds numeric, policy, and date contradictions
- **Staleness Checker**: Identifies expired/outdated documents
- **Gap Analyzer**: Discovers coverage gaps across documents

### 6. Output Formats

- **Alerts**: Indexed in Elasticsearch, sent to Slack
- **Reports**: Markdown, Excel, PDF exports
- **Dashboard**: Real-time metrics and visualizations
