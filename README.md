# Syllabus Chatbot

This project is a simple retrieval-augmented generation (RAG) chatbot for answering questions about a syllabus PDF. It reads the PDF, splits it into smaller chunks, creates vector embeddings, stores them in a local Chroma database, and uses a Groq language model to answer questions based on the retrieved context.

## What it does

- Loads a syllabus PDF
- Splits the document into chunks
- Creates embeddings with Hugging Face
- Stores and searches embeddings locally with Chroma
- Answers user questions using a Groq LLM

## Requirements

- Python 3.10+
- A Groq API key
- The syllabus PDF file

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a dotenv file named .env and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. Place your syllabus PDF in the project folder. The default file name is gcis_syllabus.pdf. If you use a different name, update the PDF path in Chatbot.py.

5. Run the chatbot:
   ```bash
   python Chatbot.py
   ```

## Project structure

- Chatbot.py: Main application logic
- requirements.txt: Python dependencies
- chroma_db_store/: Local Chroma vector database storage
- gcis_syllabus.pdf: Example syllabus source file

## How to use it

When the program starts, it will:

- Build the vector database if it does not already exist
- Load the existing database if it is already present
- Prompt you with a question input

Type exit or quit to stop the program.

### Useful command-line options

You can customize the run with flags such as:

```bash
python Chatbot.py --pdf gcis_syllabus.pdf --retrieval-k 3 --chunk-size 500 --chunk-overlap 50 --reset-db
```

### Run the Streamlit app

For a browser-based interface, start:

```bash
streamlit run app.py
```

You can also configure values through environment variables such as:

- PDF_FILE_PATH
- CHROMA_DB_DIR
- EMBEDDING_MODEL
- CHUNK_SIZE
- CHUNK_OVERLAP
- RETRIEVAL_K
- GROQ_API_KEY

The chatbot now also shows a short list of source references for each answer.

## Ways to improve this project

Here are some strong next steps for making this project more useful:

- Add a web interface using Streamlit or Gradio
- Add conversation memory so the bot can remember previous questions
- Support multiple PDFs or whole course folders
- Improve source citations with page-level excerpts
- Add better logging and a simple debug mode
- Add tests for the indexing and chat flow
- Improve retrieval quality with reranking or hybrid search

## Notes

- The first run may take a little longer because the vector database is being created.
- If the syllabus file is missing, the program will stop with an error message.
# rag_chatbot
