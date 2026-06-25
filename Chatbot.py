import argparse
import os
import shutil
from pathlib import Path

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

DEFAULT_PDF_FILE_PATH = "gcis_syllabus.pdf"
DEFAULT_CHROMA_DB_DIR = "./chroma_db_store"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVAL_K = 3


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about a syllabus PDF using a RAG chatbot")
    parser.add_argument("--pdf", default=os.getenv("PDF_FILE_PATH", DEFAULT_PDF_FILE_PATH), help="Path to the syllabus PDF")
    parser.add_argument("--db-dir", default=os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DB_DIR), help="Directory for the Chroma vector store")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL), help="Hugging Face embedding model")
    parser.add_argument("--chunk-size", type=int, default=get_int_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE), help="Chunk size for document splitting")
    parser.add_argument("--chunk-overlap", type=int, default=get_int_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP), help="Chunk overlap for document splitting")
    parser.add_argument("--retrieval-k", type=int, default=get_int_env("RETRIEVAL_K", DEFAULT_RETRIEVAL_K), help="Number of relevant chunks to retrieve")
    parser.add_argument("--reset-db", action="store_true", help="Rebuild the Chroma vector store from scratch")
    return parser.parse_args()


def build_vector_store(pdf_path: str, db_dir: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> Chroma:
    print("Reading and processing syllabus PDF...")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Could not find '{pdf_path}'. Please provide a valid PDF path.")

    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)

    print(f"Successfully split syllabus into {len(chunks)} chunks.")

    db_path = Path(db_dir)
    if db_path.exists() and db_path.is_dir():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    print("Creating your local Chroma database store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path),
    )
    print("Chroma DB configuration successfully saved!")
    return vector_store


def load_vector_store(pdf_path: str, db_dir: str, embedding_model: str) -> Chroma:
    db_path = Path(db_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if db_path.exists() and any(db_path.iterdir()):
        print("Existing Chroma database folder found. Loading data configuration instantly...")
        return Chroma(persist_directory=str(db_path), embedding_function=embeddings)

    return build_vector_store(pdf_path, db_dir, embedding_model, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


def format_sources(documents) -> list[str]:
    sources = []
    seen = set()

    for doc in documents:
        raw_source = doc.metadata.get("source") or "unknown source"
        source_name = Path(raw_source).name or raw_source
        page = doc.metadata.get("page")
        label = f"{source_name}"
        if page is not None:
            label += f" (page {page + 1})"

        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources[:5]


def run_chatbot(vector_store: Chroma, retrieval_k: int, embedding_model: str) -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to your .env file before running the chatbot.")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

    system_prompt = (
        "You are a helpful university academic assistant.\n"
        "Use the provided syllabus context fragments to answer the student's question accurately.\n"
        "If you do not know the answer based on the context, state clearly that the "
        "syllabus does not explicitly provide that information.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nSyllabus Assistant Active! Type 'exit' to quit.")
    print("-----------------------------------------------------")

    while True:
        user_query = input("\nStudent Question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_query.strip():
            continue

        print("Searching syllabus and formulating answer...")
        response = rag_chain.invoke({"input": user_query})
        relevant_docs = retriever.invoke(user_query)
        source_refs = format_sources(relevant_docs)

        print("\nAssistant Response:")
        print(response["answer"])

        if source_refs:
            print("\nSources:")
            for ref in source_refs:
                print(f"- {ref}")


def main() -> None:
    args = parse_args()

    if args.reset_db:
        print("Resetting vector store...")

    try:
        if args.reset_db or not Path(args.db_dir).exists() or not any(Path(args.db_dir).iterdir()):
            db = build_vector_store(args.pdf, args.db_dir, args.embedding_model, args.chunk_size, args.chunk_overlap)
        else:
            db = load_vector_store(args.pdf, args.db_dir, args.embedding_model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    run_chatbot(db, args.retrieval_k, args.embedding_model)


if __name__ == "__main__":
    main()