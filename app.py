import os
import shutil
import sqlite3
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
import streamlit as st


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from torch import embedding

DB_PATH = "rag_chat.db"

load_dotenv()

def get_llm(model_name: str, *, temperature: float = 0.1, max_tokens: int = 512) -> ChatOpenAI:
    api_key = st.secrets["OPENROUTER_API_KEY"] or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error(
            "Missing `OPENROUTER_API_KEY` environment variable. "
            "Please set it before running the app."
        )
        st.stop()

    # Optional but recommended headers for OpenRouter rankings/analytics
    referer = os.getenv("OPENROUTER_SITE_URL", "")
    title = os.getenv("OPENROUTER_SITE_NAME", "Streamlit RAG Chatbot")

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": title,
        },
        temperature=temperature,
        max_tokens=max_tokens,  # keep responses short to avoid credit/max_token issues
    )


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            vector_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )
    conn.commit()
    conn.close()


def get_or_create_user(username: str) -> Optional[int]:
    if not username:
        return None
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
    conn.commit()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return int(row["id"]) if row else None


def list_conversations(user_id: int) -> List[sqlite3.Row]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, created_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def create_conversation(user_id: int, title: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user_id, title or "New chat"),
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return int(conv_id)


def update_conversation_title(conversation_id: int, title: str) -> None:
    if not title:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (title, conversation_id),
    )
    conn.commit()
    conn.close()


def save_message(conversation_id: int, role: str, content: str) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content),
    )
    conn.commit()
    conn.close()


def load_messages(conversation_id: int) -> List[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (conversation_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def update_conversation_vector_path(conversation_id: int, vector_path: str) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET vector_path = ? WHERE id = ?",
        (vector_path, conversation_id),
    )
    conn.commit()
    conn.close()


def get_conversation_vector_path(conversation_id: int) -> Optional[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT vector_path FROM conversations WHERE id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row and row["vector_path"]:
        return str(row["vector_path"])
    return None


def delete_conversation(conversation_id: int) -> None:
    """Delete a conversation, its messages, and any associated vector store directory."""
    # First fetch vector path (if any)
    vector_path = get_conversation_vector_path(conversation_id)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

    # Remove saved vector store directory if it exists
    if vector_path and os.path.isdir(vector_path):
        try:
            shutil.rmtree(vector_path)
        except OSError:
            pass


def load_uploaded_file(uploaded_file) -> List[Document]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Persist to a temporary file so standard loaders can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif suffix in {".csv"}:
            loader = CSVLoader(tmp_path, encoding="utf-8")
        else:
            st.warning(f"Unsupported file type for {uploaded_file.name}; skipping.")
            return []

        return loader.load()
    finally:
        # Do not leave temp files lying around
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_documents(
    uploaded_files: List, website_url: Optional[str]
) -> List[Document]:
    docs: List[Document] = []

    for f in uploaded_files:
        docs.extend(load_uploaded_file(f))

    if website_url:
        try:
            web_loader = WebBaseLoader(website_url)
            docs.extend(web_loader.load())
        except Exception as e:
            st.warning(f"Failed to load website content: {e}")

    return docs


def build_vectorstore(docs: List[Document]) -> Optional[FAISS]:
    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if docs and docs[0].metadata.get("source", "").endswith(".csv"):
        return FAISS.from_documents(docs, embeddings)

    # Otherwise split (PDF, TXT, etc.)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)

    return FAISS.from_documents(split_docs, embeddings)


def format_chat_context(chat_history: List[dict], max_turns: int = 8) -> str:
    """Return a short textual representation of recent conversation turns."""
    if not chat_history:
        return ""

    turns = []
    # Take last max_turns messages (user + assistant)
    for msg in chat_history[-max_turns:]:
        role = msg.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        turns.append(f"{prefix}: {msg.get('content', '')}")
    return "\n".join(turns)


def _fallback_title_from_text(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if not words:
        return "New chat"
    return " ".join(words[:6])[:60].strip()


def generate_conversation_title(model_name: str, chat_history: List[dict]) -> str:
    # Prefer the first user message as the topic seed.
    first_user = next((m.get("content", "") for m in chat_history if m.get("role") == "user"), "")
    seed = (first_user or "").strip()
    if not seed:
        return "New chat"

    llm = get_llm(model_name, temperature=0.2, max_tokens=24)
    prompt = (
        "Create a short ChatGPT-style conversation title (3-6 words) based on the user's questions.\n"
        "Rules:\n"
        "- Return ONLY the title.\n"
        "- No quotes.\n"
        "- No trailing period.\n\n"
        f"User topic:\n{seed}\n\n"
        "Title:"
    )
    try:
        resp = llm.invoke(prompt)
        title = getattr(resp, "content", str(resp)).strip()
    except Exception:
        title = ""

    title = (title or "").strip().strip('"').strip("'")
    title = title.splitlines()[0].strip() if title else ""
    if title.endswith("."):
        title = title[:-1].strip()
    title = title[:80].strip()
    return title or _fallback_title_from_text(seed)


def rewrite_query_with_history(
    llm: ChatOpenAI, query: str, chat_history: Optional[List[dict]]
) -> str:
    """Use the LLM to turn a follow-up question into a standalone query."""
    if not chat_history:
        return query

    history_text = format_chat_context(chat_history, max_turns=8)
    if not history_text:
        return query

    rewrite_prompt = (
        "You are helping a retrieval system understand conversational questions.\n"
        "Given the conversation so far and the user's latest question, rewrite the "
        "question so it can be understood without the conversation.\n"
        "Do not answer the question, only rewrite it.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user question: {query}\n\n"
        "Standalone rewritten question:"
    )
    try:
        resp = llm.invoke(rewrite_prompt)
        rewritten = getattr(resp, "content", str(resp)).strip()
        return rewritten or query
    except Exception:
        # If anything goes wrong, fall back to the original query
        return query


def run_rag_query(
    vectorstore: FAISS,
    model_name: str,
    query: str,
    chat_history: Optional[List[dict]] = None,
) -> tuple[str, List[Document]]:
    llm = get_llm(model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    history_text = format_chat_context(chat_history, max_turns=6)
    effective_query = f"{history_text}\n\nCurrent question: {query}"
    source_docs = retriever.invoke(effective_query) 

    context = "\n\n".join(doc.page_content for doc in source_docs)
    history_text = format_chat_context(chat_history or [])
    prompt_parts = [
        "You are a helpful assistant for a Retrieval-Augmented Generation (RAG) chatbot.",
        "Use the information from the document provided to answer the user's question.",
        "If the answer is not contained in the context, say that you don't know.",
        "If the question is a follow-up, use the conversation history to resolve references like 'it', 'they', 'that', etc.",
    ]
    prompt = "\n".join(prompt_parts) + "\n\n"
    if history_text:
        prompt += f"Conversation so far:\n{history_text}\n\n"
    if effective_query != query:
        prompt += f"Standalone interpretation of the question: {effective_query}\n\n"
    prompt += f"Retrieved context:\n{context}\n\nOriginal user question: {query}"

    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))
    return answer, source_docs


def init_session_state() -> None:
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "rag_model_name" not in st.session_state:
        st.session_state["rag_model_name"] = "openai/gpt-4.1-mini"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = None
    if "conversation_title" not in st.session_state:
        st.session_state["conversation_title"] = ""


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="💬",
        layout="wide",
    )

    init_db()
    init_session_state()

    st.title("RAG Chatbot with LangChain")
    st.markdown(
        "Upload your documents or provide a website URL, then ask questions."
    )

    with st.sidebar:
        st.header("Account")
        username = st.text_input(
            "Username",
            value=st.session_state.get("username", ""),
            help="Used to separate and load your chats, similar to ChatGPT.",
        ).strip()
        st.session_state["username"] = username

        user_id = get_or_create_user(username) if username else None
        st.session_state["user_id"] = user_id

        if not user_id:
            st.info("Enter a username to start or load chats.")
        else:
            st.markdown("### Your chats")
            conv_rows = list_conversations(user_id)
            # Use an ID-suffixed label to avoid collisions when titles repeat.
            conv_options = {f"{row['title']} · {row['id']}": row for row in conv_rows}
            selected_label = st.selectbox(
                "Select a chat",
                ["New chat"] + list(conv_options.keys()),
                index=0,
            )

            if selected_label == "+ New chat":
                if st.button("Start new chat"):
                    st.session_state["conversation_id"] = None
                    st.session_state["conversation_title"] = ""
                    st.session_state["chat_history"] = []
                    st.session_state["vectorstore"] = None
                    st.session_state["last_sources"] = []
                    st.success("Started a new chat. Build a knowledge base to begin.")
            else:
                selected_row = conv_options.get(selected_label)
                load_col, delete_col = st.columns(2)
                if selected_row is not None and load_col.button("Load selected chat"):
                    conv_id = int(selected_row["id"])
                    st.session_state["conversation_id"] = conv_id
                    st.session_state["conversation_title"] = selected_row["title"]
                    st.session_state["chat_history"] = load_messages(conv_id)

                    # Reload vector store for this conversation if available
                    vector_path = get_conversation_vector_path(conv_id)
                    if vector_path and os.path.isdir(vector_path):
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            vs = FAISS.load_local(
                                vector_path,
                                embeddings,
                                allow_dangerous_deserialization=True,
                            )
                            st.session_state["vectorstore"] = vs
                        except Exception:
                            st.session_state["vectorstore"] = None
                    else:
                        st.session_state["vectorstore"] = None

                    st.success(f"Loaded chat: {selected_row['title']}")

                if selected_row is not None and delete_col.button("Delete selected chat"):
                    conv_id = int(selected_row["id"])
                    delete_conversation(conv_id)

                    # If this was the active chat, clear current state
                    if st.session_state.get("conversation_id") == conv_id:
                        st.session_state["conversation_id"] = None
                        st.session_state["conversation_title"] = ""
                        st.session_state["chat_history"] = []
                        st.session_state["vectorstore"] = None
                        st.session_state["last_sources"] = []

                    st.success(f"Deleted chat: {selected_row['title']}")
                    st.rerun()

            st.markdown("---")
            st.header("Knowledge base")

            st.markdown("### Data sources")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "md", "csv"],
                accept_multiple_files=True,
            )
            website_url = st.text_input(
                "Website URL (optional)",
                placeholder="https://example.com/article",
            )
            if st.button("Build / update knowledge base"):
                with st.spinner("Processing documents and building vector store..."):
                    docs = build_documents(uploaded_files or [], website_url.strip() or None)
                    vectorstore = build_vectorstore(docs)

                    if vectorstore is None:
                        st.error("No valid content found. Please upload files or provide a URL.")
                    else:
                        # Ensure we have a conversation row to associate this KB with
                        if st.session_state["conversation_id"] is None:
                            title = st.session_state.get("conversation_title") or "New chat"
                            conv_id = create_conversation(user_id, title)
                            st.session_state["conversation_id"] = conv_id
                        conv_id = st.session_state["conversation_id"]

                        # Save vector store locally per conversation
                        vector_dir = os.path.join("vectorstores", f"user_{user_id}_conv_{conv_id}")
                        os.makedirs(vector_dir, exist_ok=True)
                        vectorstore.save_local(vector_dir)
                        update_conversation_vector_path(conv_id, vector_dir)

                        st.session_state["vectorstore"] = vectorstore
                        st.success(
                            f"Knowledge base built from {len(docs)} document chunks. You can now start chatting."
                        )

            if st.button("Clear chat & knowledge base"):
                st.session_state["vectorstore"] = None
                st.session_state["chat_history"] = []
                st.session_state["last_sources"] = []
                st.session_state["conversation_id"] = None
                st.session_state["conversation_title"] = ""
                st.session_state["uploaded_files"] = None
                st.success("Cleared current chat state. Select or start a chat to continue.")



    col_chat, col_sources = st.columns([2, 1])

    with col_chat:
        st.subheader("Chat")

        # Always render existing history so loaded conversations are visible,
        # even if no knowledge base is currently built.
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state["vectorstore"] is None:
            st.info("Build the knowledge base in the sidebar to start asking questions.")

        # Always render the input so it stays at the bottom like ChatGPT.
        user_input = st.chat_input("Ask a question about your data...")
        if user_input:
            # Ensure there is a conversation row for this user
            if st.session_state["user_id"] and st.session_state["conversation_id"] is None:
                title = st.session_state.get("conversation_title") or "New chat"
                conv_id = create_conversation(st.session_state["user_id"], title)
                st.session_state["conversation_id"] = conv_id

            conv_id = st.session_state.get("conversation_id")

            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )
            if conv_id:
                save_message(conv_id, "user", user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state["vectorstore"] is None:
                msg = "Please build the knowledge base from the sidebar first."
                with st.chat_message("assistant"):
                    st.markdown(msg)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": msg}
                )
                if conv_id:
                    save_message(conv_id, "assistant", msg)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        vectorstore: FAISS = st.session_state["vectorstore"]
                        model_name = st.session_state.get("rag_model_name") or "openai/gpt-4.1-mini"
                        answer, sources = run_rag_query(
                            vectorstore=vectorstore,
                            model_name=model_name,
                            query=user_input,
                            chat_history=st.session_state.get("chat_history", []),
                        )
                        st.markdown(answer)
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": answer}
                        )
                        if conv_id:
                            save_message(conv_id, "assistant", answer)

                        # Auto-generate a ChatGPT-like title once, for new/default chats.
                        current_title = (st.session_state.get("conversation_title") or "").strip()
                        if conv_id and (not current_title or current_title.lower() == "new chat"):
                            new_title = generate_conversation_title(model_name, st.session_state["chat_history"])
                            update_conversation_title(conv_id, new_title)
                            st.session_state["conversation_title"] = new_title
                            st.rerun()

                # Store sources for display in the right column
                st.session_state["last_sources"] = sources

    with col_sources:
        st.subheader("Sources")
        sources: List[Document] = st.session_state.get("last_sources", [])
        if not sources:
            st.write("Ask a question to see which chunks were used.")
        else:
            for i, doc in enumerate(sources, start=1):
                st.markdown(f"**Source {i}**")
                st.markdown(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
                meta_str = ", ".join(f"{k}: {v}" for k, v in (doc.metadata or {}).items())
                if meta_str:
                    st.caption(meta_str)
                st.markdown("---")


if __name__ == "__main__":
    main()
