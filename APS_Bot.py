"""Local APS Tamil School FAQ bot.

Workflow: retrieve -> grade -> answer OR escalate to human -> end.

Install once:
    python -m pip install -U langchain-chroma langchain-huggingface sentence-transformers langgraph

Run interactively:
    python APS_Bot.py

Or ask one question:
    python APS_Bot.py "How do I report an absence?"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
COLLECTION_NAME = "online_tamil_school_local"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Increase this value to escalate more questions; decrease it to answer more.
RELEVANCE_THRESHOLD = 0.35


class BotState(TypedDict, total=False):
    question: str
    documents: list[Document]
    best_score: float
    status: Literal["answered", "escalated"]
    answer: str


def get_vector_store() -> Chroma:
    """Connect to the local vector store created by ingest_data.py."""
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            "The Chroma database was not found. Run 'python ingest_data.py' first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )


VECTOR_STORE = get_vector_store()


def retrieve(state: BotState) -> BotState:
    """Retrieve the three most relevant school records."""
    results = VECTOR_STORE.similarity_search_with_relevance_scores(
        state["question"], k=3
    )
    return {
        "documents": [document for document, _ in results],
        "best_score": results[0][1] if results else 0.0,
    }


def grade_documents(state: BotState) -> Literal["answer", "escalate"]:
    """Grade whether the best retrieved record is relevant enough to answer."""
    if state.get("documents") and state.get("best_score", 0.0) >= RELEVANCE_THRESHOLD:
        return "answer"
    return "escalate"


def answer(state: BotState) -> BotState:
    """Give a response grounded only in the highest-ranked school record."""
    best_document = state["documents"][0]
    title = best_document.metadata.get("title", "School information")
    source = best_document.metadata.get("source", "School knowledge base")
    return {
        "status": "answered",
        "answer": (
            f"{title}\n\n{best_document.page_content.replace(f'Title: {title}\n\n', '')}"
            f"\n\nSource: {source}"
        ),
    }


def escalate_to_human(_: BotState) -> BotState:
    """Route questions without a reliable knowledge-base match to staff."""
    return {
        "status": "escalated",
        "answer": (
            "I do not have enough reliable school information to answer that. "
            "Your question has been escalated to a human school administrator."
        ),
    }


def build_bot():
    workflow = StateGraph(BotState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    workflow.add_node("escalate", escalate_to_human)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"answer": "answer", "escalate": "escalate"},
    )
    workflow.add_edge("answer", END)
    workflow.add_edge("escalate", END)
    return workflow.compile()


BOT = build_bot()


def ask(question: str) -> BotState:
    """Run one question through the retrieve-grade-answer/escalate workflow."""
    return BOT.invoke({"question": question.strip()})


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Ask APS Tamil School Bot a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = ask(question)
    print(f"\nStatus: {result['status']}")
    print(f"Best relevance score: {result.get('best_score', 0.0):.2f}\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
