"""Hybrid retrieval for the APS Online Tamil School knowledge base.

The retriever combines:
* Semantic search from the persistent Chroma database (handles paraphrased questions)
* BM25 keyword search over the documents stored in that database (handles exact terms)
* Reciprocal Rank Fusion (RRF) to merge both result lists

Install once:
    python -m pip install -U langchain-community rank-bm25 numpy

Run after ingest_data.py has created ./chroma_db:
    python retriever.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
COLLECTION_NAME = "online_tamil_school_local"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
RRF_K = 60


@dataclass
class HybridRetriever:
    """Hybrid Chroma + BM25 retriever backed by the persisted knowledge base."""

    vectorstore: Chroma
    bm25: BM25Retriever
    embeddings: HuggingFaceEmbeddings

    def semantic_search(self, query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
        """Return documents and Chroma relevance scores for a semantic-only search."""
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """Return hybrid results merged with weighted Reciprocal Rank Fusion."""
        semantic_docs = [document for document, _ in self.semantic_search(query, k=k)]
        keyword_docs = self.bm25.invoke(query)[:k]

        fused_scores: dict[str, float] = {}
        fused_docs: dict[str, Document] = {}
        for weight, results in (
            (SEMANTIC_WEIGHT, semantic_docs),
            (BM25_WEIGHT, keyword_docs),
        ):
            for rank, document in enumerate(results, start=1):
                key = self._document_key(document)
                fused_docs[key] = document
                fused_scores[key] = fused_scores.get(key, 0.0) + weight / (RRF_K + rank)

        ordered_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [fused_docs[key] for key in ordered_keys[:k]]

    def score_query(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        """Pair each document with cosine similarity to the query.

        RRF values in ``retrieve`` are rank-based. These cosine similarities are
        a separate, interpretable signal that can be used by a confidence gate.
        """
        if not docs:
            return []

        query_vector = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        query_vector /= np.linalg.norm(query_vector) + 1e-12

        document_vectors = np.array(
            self.embeddings.embed_documents([document.page_content for document in docs]),
            dtype=np.float32,
        )
        document_vectors /= np.linalg.norm(
            document_vectors, axis=1, keepdims=True
        ) + 1e-12

        similarities = document_vectors @ query_vector
        return list(zip(docs, similarities.tolist()))

    @staticmethod
    def _document_key(document: Document) -> str:
        """Use the persisted record ID when available; otherwise use its text."""
        return str(document.metadata.get("id", document.page_content))


def load_documents_from_db(vectorstore: Chroma) -> list[Document]:
    """Read every stored document and its metadata from the Chroma collection."""
    stored = vectorstore.get(include=["documents", "metadatas"])
    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    ids = stored.get("ids") or []

    documents = []
    for index, text in enumerate(texts):
        metadata = dict(metadatas[index] or {})
        metadata.setdefault("id", ids[index])
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_retriever() -> HybridRetriever:
    """Open the persistent Chroma DB and construct its hybrid retriever."""
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            "The Chroma database was not found. Run 'python ingest_data.py' first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )
    documents = load_documents_from_db(vectorstore)
    if not documents:
        raise ValueError("The Chroma collection is empty. Run 'python ingest_data.py' first.")

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = TOP_K
    return HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    retriever = build_retriever()
    sample_queries = [
        "How do I report that my child will miss class?",
        "Can my child join if they do not know Tamil?",
        "I cannot hear the teacher during the online lesson.",
    ]

    for query in sample_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)
        for document, similarity in retriever.score_query(query, results):
            source = document.metadata.get("source", "Unknown")
            title = document.metadata.get("title", "Untitled")
            print(f"  [{similarity:.3f}] {source} :: {title}")
