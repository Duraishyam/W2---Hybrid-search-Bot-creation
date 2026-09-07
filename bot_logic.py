"""Shared retrieval, answer, and service-request routing logic for APS Bot."""

from __future__ import annotations

import re

from retriever import HybridRetriever


ANSWER_DOCUMENT_TYPES = {"faq", "manual"}
SERVICE_DOCUMENT_TYPES = {"service_request"}
CONFIDENCE_THRESHOLD = 0.50
SERVICE_REQUEST_THRESHOLD = 0.50

# The school's knowledge base is Tamil-only. These requests must not be answered
# with a loosely related Tamil-school record.
UNSUPPORTED_LANGUAGE_TERMS = {"hindi", "telugu", "sanskrit", "malayalam", "kannada"}

SERVICE_REQUEST_PATTERN = re.compile(
    r"\b("
    r"i need|please|recurring conflict|switch class|change class|"
    r"cannot open|can't open|cannot join|can't join|"
    r"send me a receipt|need a receipt|report an absence|"
    r"update my contact|withdraw my child|request a refund|"
    r"progress update|progress report|update on .*progress|"
    r"enrollment information|placement assessment|"
    r"homework or classwork|fee receipt|invoice|payment confirmation|"
    r"classroom concern"
    r")\b",
    re.IGNORECASE,
)


def is_service_request(question: str) -> bool:
    """Identify direct parent actions that require a school staff workflow."""
    return bool(SERVICE_REQUEST_PATTERN.search(question))


def has_unsupported_language_request(question: str) -> bool:
    """Prevent a Tamil-school result from answering another-language questions."""
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    return bool(tokens & UNSUPPORTED_LANGUAGE_TERMS)


def run_bot(retriever: HybridRetriever, question: str) -> dict:
    """Answer FAQ/manual questions or route direct service requests to staff.

    The first document returned by ``HybridRetriever.retrieve`` is the top
    Reciprocal Rank Fusion result. Its cosine score is used only as the
    confidence gate; it does not replace the hybrid ranking.
    """
    if has_unsupported_language_request(question):
        return {
            "decision": "escalate",
            "scored": [],
            "top_sim": 0.0,
            "answer": (
                "I do not have enough reliable school information to answer that. "
                "This question needs help from a school administrator."
            ),
            "route_type": "out_of_scope",
        }

    answer_documents = retriever.retrieve(
        question,
        document_types=ANSWER_DOCUMENT_TYPES,
    )
    answer_scored = sorted(
        retriever.score_query(question, answer_documents),
        key=lambda item: item[1],
        reverse=True,
    )

    if is_service_request(question):
        service_documents = retriever.retrieve(
            question,
            document_types=SERVICE_DOCUMENT_TYPES,
        )
        service_scored = sorted(
            retriever.score_query(question, service_documents),
            key=lambda item: item[1],
            reverse=True,
        )
        if service_scored:
            service_document, service_score = service_scored[0]
            return {
                "decision": "answer",
                "scored": service_scored,
                "top_sim": service_score,
                "answer": service_document.page_content,
                "route_type": "service_request",
            }

    if answer_scored and answer_scored[0][1] >= CONFIDENCE_THRESHOLD:
        answer_document, answer_score = answer_scored[0]
        return {
            "decision": "answer",
            "scored": answer_scored,
            "top_sim": answer_score,
            "answer": answer_document.page_content,
            "route_type": "knowledge_base",
        }

    return {
        "decision": "escalate",
        "scored": answer_scored,
        "top_sim": answer_scored[0][1] if answer_scored else 0.0,
        "answer": (
            "I do not have enough reliable school information to answer that. "
            "This question needs help from a school administrator."
        ),
        "route_type": "out_of_scope",
    }
