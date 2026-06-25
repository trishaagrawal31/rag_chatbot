import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_PDF_PATH = os.getenv("PDF_FILE_PATH") or "gcis_syllabus.pdf"
DEFAULT_DB_DIR = os.getenv("CHROMA_DB_DIR") or "./chroma_db_store"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or "500")
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or "50")
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K") or "3")


@st.cache_resource(show_spinner=False)
def build_or_load_store(pdf_path: str, db_dir: str, embedding_model: str, chunk_size: int, chunk_overlap: int):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Could not find '{pdf_path}'.")

    db_path = Path(db_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if db_path.exists() and any(db_path.iterdir()):
        return Chroma(persist_directory=str(db_path), embedding_function=embeddings)

    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path),
    )


@st.cache_data(show_spinner=False)
def format_sources(documents):
    sources = []
    seen = set()

    for doc in documents:
        raw_source = doc.metadata.get("source") or "unknown source"
        source_name = Path(raw_source).name or raw_source
        page = doc.metadata.get("page")
        label = source_name
        if page is not None:
            label += f" (page {page + 1})"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources[:5]


def main() -> None:
    st.set_page_config(page_title="Syllabus Chatbot", page_icon="📘", layout="centered")
    st.title("Syllabus Chatbot")
    st.caption("Ask questions about your syllabus using a local RAG workflow.")

    with st.sidebar:
        st.header("Settings")
        pdf_path = st.text_input("PDF path", value=DEFAULT_PDF_PATH)
        db_dir = st.text_input("Chroma DB directory", value=DEFAULT_DB_DIR)
        embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=10)
        retrieval_k = st.number_input("Retrieval results", min_value=1, max_value=10, value=DEFAULT_RETRIEVAL_K, step=1)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        vector_store = build_or_load_store(pdf_path, db_dir, embedding_model, int(chunk_size), int(chunk_overlap))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY. Add it to your .env file before chatting.")
        st.stop()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

    system_prompt = (
        "You are a helpful university academic assistant.\n"
        "Use the provided syllabus context fragments to answer the student's question accurately.\n"
        "If you do not know the answer based on the context, state clearly that the syllabus does not explicitly provide that information.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    prompt_text = st.chat_input("Ask about the syllabus")
    if prompt_text:
        st.session_state.chat_history.append({"role": "user", "content": prompt_text, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Searching the syllabus..."):
                response = rag_chain.invoke({"input": prompt_text})
                relevant_docs = retriever.invoke(prompt_text)
                sources = format_sources(relevant_docs)
            st.markdown(response["answer"])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": sources,
            })


if __name__ == "__main__":
    main()
