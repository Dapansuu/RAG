import os
import json
import tempfile
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

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


load_dotenv()

CHAT_STORE_PATH = "chat_history.json"


def get_llm(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
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
        temperature=0.1,
        max_tokens=512,  # keep responses short to avoid credit/max_token issues
    )


def load_chat_store() -> dict:
    if not os.path.exists(CHAT_STORE_PATH):
        return {}
    try:
        with open(CHAT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_chat_store(store: dict) -> None:
    with open(CHAT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Use a rewritten standalone query for retrieval so follow-ups work better.
    effective_query = rewrite_query_with_history(llm, query, chat_history)

    # In recent LangChain versions, retrievers are Runnables; use invoke() instead of get_relevant_documents()
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
        st.session_state["user_id"] = ""
    if "conversation_title" not in st.session_state:
        st.session_state["conversation_title"] = ""


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="💬",
        layout="wide",
    )

    init_session_state()

    st.title("RAG Chatbot with LangChain")
    st.markdown(
        "Upload your documents or provide a website URL, then ask questions."
    )

    with st.sidebar:
        st.header("User & conversations")
        user_id = st.text_input(
            "User ID",
            value=st.session_state.get("user_id", ""),
            help="Used to save and load your chats separately from other users.",
        ).strip()
        st.session_state["user_id"] = user_id

        store = load_chat_store()
        user_chats = sorted(store.get(user_id, {}).keys()) if user_id else []
        selected_chat = st.selectbox(
            "Load saved chat",
            ["<New chat>"] + user_chats,
            index=0,
        )

        if user_id and selected_chat != "<New chat>" and st.button("Load selected chat"):
            convo = store[user_id][selected_chat]
            st.session_state["conversation_title"] = selected_chat
            st.session_state["chat_history"] = convo.get("messages", [])
            st.success(f"Loaded chat: {selected_chat}")

        conv_title = st.text_input(
            "Conversation title",
            value=st.session_state.get("conversation_title", ""),
            help="Name for this chat when saving.",
        )
        st.session_state["conversation_title"] = conv_title

        if user_id and conv_title and st.button("Save current chat"):
            store = load_chat_store()
            store.setdefault(user_id, {})
            store[user_id][conv_title] = {
                "messages": st.session_state.get("chat_history", []),
            }
            save_chat_store(store)
            st.success("Chat saved.")

        st.markdown("---")
        st.header("Configuration")

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
        if st.button("Build knowledge base"):
            with st.spinner("Processing documents and building vector store..."):
                docs = build_documents(uploaded_files or [], website_url.strip() or None)
                vectorstore = build_vectorstore(docs)

                if vectorstore is None:
                    st.error("No valid content found. Please upload files or provide a URL.")
                else:
                    st.session_state["vectorstore"] = vectorstore
                    # keep current model name in session; default is set in init_session_state
                    st.success(
                        f"Knowledge base built from {len(docs)} document chunks. You can now start chatting."
                    )
        if st.button("Clear chat & knowledge base"):
            st.session_state["vectorstore"] = None
            st.session_state["rag_model_name"] = None
            st.session_state["chat_history"] = []
            st.session_state["last_sources"] = []
            st.success("Cleared chat history and knowledge base.")



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
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state["vectorstore"] is None:
                msg = "Please build the knowledge base from the sidebar first."
                with st.chat_message("assistant"):
                    st.markdown(msg)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": msg}
                )
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
