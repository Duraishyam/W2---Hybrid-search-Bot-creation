# APS Online Tamil School — Retrieval Architecture

```mermaid
flowchart TB
    subgraph Sources[School knowledge sources]
        FAQ[online_tamil_school_faq.json]
        Manual[manual.json]
        Requests[service_request.json]
    end

    subgraph Ingestion[Offline ingestion]
        Ingest[ingest_data.py]
        Embed[Hugging Face embeddings<br/>all-MiniLM-L6-v2]
        Chroma[(ChromaDB<br/>chroma_db)]
    end

    FAQ --> Ingest
    Manual --> Ingest
    Requests --> Ingest
    Ingest --> Embed --> Chroma

    subgraph Runtime[Runtime — Streamlit app]
        User[Parent question]
        App[streamlit_app.py]
        Retriever[retriever.py<br/>HybridRetriever]

        Semantic[Semantic search<br/>Chroma vector similarity]
        LoadDocs[Load stored documents<br/>from ChromaDB]
        BM25[BM25 keyword search]
        RRF[Weighted RRF fusion<br/>60% semantic / 40% BM25]
        Score[Cosine-similarity<br/>confidence score]
        Gate{Score >= 0.35?}
        Answer[Return highest-ranked<br/>school record]
        Escalate[Flag for human<br/>school administrator]
    end

    User --> App --> Retriever
    Retriever --> Semantic
    Chroma --> Semantic
    Retriever --> LoadDocs
    Chroma --> LoadDocs --> BM25
    Semantic --> RRF
    BM25 --> RRF --> Score --> Gate
    Gate -- Yes --> Answer
    Gate -- No --> Escalate
    Answer --> App
    Escalate --> App

    subgraph DirectBot[Optional command-line workflow]
        APSBot[APS_Bot.py]
        DirectSemantic[Direct Chroma semantic search]
    end

    APSBot --> DirectSemantic
    Chroma --> DirectSemantic
```

## Retrieval mechanisms

1. **Semantic search** uses ChromaDB and local Hugging Face embeddings to find records with similar meaning.
2. **BM25 search** uses keyword matching across documents loaded from ChromaDB.
3. **Reciprocal Rank Fusion (RRF)** combines the semantic and BM25 result rankings.
4. **Cosine similarity scoring** determines whether the app answers from the best record or flags the question for a human administrator.
