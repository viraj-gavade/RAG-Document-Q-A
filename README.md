# RAG Document Q&A

A Streamlit-based application for Question & Answering over research papers using Retrieval-Augmented Generation (RAG) and LLMs.

## Features
- Upload multiple PDF research papers.
- Vector embedding creation using HuggingFace models.
- Semantic search and retrieval using FAISS.
- LLM-powered Q&A (Groq's Gemma2-9b-it).
- Document similarity search for context.

## How It Works
1. **Upload PDFs**: Add your research papers via the UI.
2. **Vectorization**: PDFs are split, embedded, and stored in a FAISS vector database.
3. **Ask Questions**: Enter queries to get answers based on the uploaded documents.
4. **Contextual Results**: View document snippets most relevant to your query.

## Setup
1. Clone this repo.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Add a `.env` file with your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the app:
   ```powershell
   streamlit run app.py
   ```

## Folder Structure
- `app.py` : Main Streamlit app.
- `requirements.txt` : Python dependencies.
- `research_papers/` : Example PDFs for testing.

## Requirements
See `requirements.txt` for all Python packages.

## Example PDFs
- `Attention.pdf`
- `LLM.pdf`

## License
MIT
