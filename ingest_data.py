"""Ingest the online Tamil school JSON knowledge base into Chroma.

Install the required packages first:
    pip install langchain-core langchain-chroma langchain-huggingface sentence-transformers

Then run:
    python ingest_data.py

The script uses a free local embedding model. It downloads the model once on
the first run, then stores embeddings in a persistent Chroma database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
# This differs from the earlier OpenAI-based collection so Chroma never mixes
# vectors with different dimensions or embedding models.
COLLECTION_NAME = "online_tamil_school_local"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_FILES = {
    "faq": BASE_DIR / "online_tamil_school_faq.json",
    "manual": BASE_DIR / "manual.json",
    "service_request": BASE_DIR / "service_request.json",
}


def load_documents() -> list[Document]:
    """Read all dataset records and convert them into LangChain Documents."""
    documents: list[Document] = []

    for document_type, file_path in DATA_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file was not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as file:
            records: Any = json.load(file)

        if not isinstance(records, list):
            raise ValueError(f"{file_path.name} must contain a JSON array.")

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Every entry in {file_path.name} must be a JSON object.")

            record_id = record.get("id")
            title = record.get("title")
            content = record.get("content")
            if not all(isinstance(value, str) and value.strip() for value in (record_id, title, content)):
                raise ValueError(
                    f"Each entry in {file_path.name} requires non-empty id, title, and content fields."
                )

            metadata = {
                "id": record_id,
                "source": record.get("source", "Unknown"),
                "title": title,
                "document_type": document_type,
                "file_name": file_path.name,
            }
            # Preserve optional fields such as category and priority.
            metadata.update(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"id", "source", "title", "content"}
                    and isinstance(value, (str, int, float, bool))
                }
            )

            page_content = f"Title: {title}\n\n{content}"
            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def ingest() -> None:
    """Embed the documents locally and upsert them into Chroma."""
    documents = load_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    vector_store.add_documents(documents, ids=[document.metadata["id"] for document in documents])
    print(
        f"Ingested {len(documents)} documents with local embeddings into "
        f"'{COLLECTION_NAME}' at {PERSIST_DIRECTORY}."
    )


if __name__ == "__main__":
    ingest()
