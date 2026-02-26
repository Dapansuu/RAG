# RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that lets users upload files or crawl a webpage, build a FAISS knowledge base, and ask grounded questions using an OpenRouter-hosted chat model.

## Features

- Multi-user chat separation via usernames stored in SQLite.
- Conversation history with create/load/delete chat flows.
- Knowledge base ingestion from:
  - PDF (`.pdf`)
  - Text/Markdown (`.txt`, `.md`)
  - CSV (`.csv`)
  - A single website URL
- FAISS vector store persistence per user conversation.
- Source chunk display for each answer.
- Automatic short conversation title generation.

## Project Structure

- `app.py` – Main Streamlit application (UI, ingestion, retrieval, chat, persistence).
- `requirements.txt` – Python dependencies.
- `rag_chat.db` – SQLite database (created/used at runtime).
- `vectorstores/` – Persisted FAISS indexes per conversation.

## Prerequisites

- Python 3.10+
- An OpenRouter API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your API key (choose one):

### Option A: Environment variable

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Option B: Streamlit secrets

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "your_api_key_here"
```

(Optional) You can also set:

- `OPENROUTER_SITE_URL`
- `OPENROUTER_SITE_NAME`

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How to Use

1. Enter a username in the sidebar.
2. Start a new chat or load an existing chat.
3. Upload one or more documents and/or provide a website URL.
4. Click **Build / update knowledge base**.
5. Ask questions in the chat input.
6. Review retrieved source chunks in the right-side **Sources** panel.

## Notes

- Vector stores are saved under `vectorstores/user_<user_id>_conv_<conversation_id>`.
- Chat messages and metadata are stored in `rag_chat.db`.
- If no relevant context is retrieved, the assistant is instructed to say it does not know.
