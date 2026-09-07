"""Streamlit chat interface for the APS Online Tamil School chatbot.

Install once:
    python -m pip install -U streamlit langchain-community rank-bm25 numpy

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from bot_logic import run_bot
from retriever import HybridRetriever, build_retriever


APP_TITLE = "APS Online Tamil School Assistant"
CONFIDENCE_THRESHOLD = 0.55


@st.cache_resource(show_spinner="Loading the school knowledge base...")
def get_retriever() -> HybridRetriever:
    """Load the local Chroma DB and hybrid retriever once per app server."""
    return build_retriever()


def clean_content(document_content: str, title: str) -> str:
    """Remove the title prefix already displayed in the chat response."""
    prefix = f"Title: {title}\n\n"
    return document_content.removeprefix(prefix)


def respond(question: str, retriever: HybridRetriever) -> tuple[str, str, list[tuple]]:
    """Retrieve, grade, and either answer or flag the question for staff."""
    result = run_bot(retriever, question)
    scored_documents = result.get("scored", [])
    if result.get("decision") != "answer":
        return result["answer"], "Escalation needed", scored_documents

    answer_document = scored_documents[0][0]
    title = answer_document.metadata.get("title", "School information")
    source = answer_document.metadata.get("source", "School knowledge base")
    answer = clean_content(answer_document.page_content, title)
    return f"{answer}\n\n_Source: {source}_", "Answered", scored_documents


def show_sources(scored_documents: list[tuple]) -> None:
    """Show transparent retrieval details without cluttering the answer."""
    if not scored_documents:
        return

    with st.expander("View retrieved sources"):
        for document, score in scored_documents:
            st.markdown(
                f"- **{document.metadata.get('title', 'Untitled')}** "
                f"({document.metadata.get('source', 'Unknown')}, relevance: {score:.2f})"
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="centered")
    st.title("📚 APS Online Tamil School Assistant")
    st.caption("Ask about enrollment, schedules, attendance, fees, Tamil classes, or technical support.")

    with st.sidebar:
        st.header("About")
        st.write("Answers are based only on the school's local FAQ, manual, and parent service requests.")
        st.write("Questions without a reliable match are flagged for a school administrator.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        retriever = get_retriever()
    except Exception as error:
        st.error(f"Unable to load the chatbot knowledge base: {error}")
        st.info("Run `python ingest_data.py` before starting this application.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    if question := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the school knowledge base..."):
                answer, status, sources = respond(question, retriever)
            if status == "Answered":
                st.success(status)
            else:
                st.warning(status)
            st.markdown(answer)
            show_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "status": status, "sources": sources}
        )


if __name__ == "__main__":
    main()
