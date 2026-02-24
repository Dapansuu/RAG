import os
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


def run_rag_query(
    vectorstore: FAISS, model_name: str, query: str
) -> tuple[str, List[Document]]:
    llm = get_llm(model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # In recent LangChain versions, retrievers are Runnables; use invoke() instead of get_relevant_documents()
    source_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in source_docs)
    prompt = (
        "You are a helpful assistant. Use ONLY the following context to answer the "
        "user's question. If the answer cannot be found in the context, say that you "
        "don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))
    return answer, source_docs


def init_session_state() -> None:
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "rag_model_name" not in st.session_state:
        st.session_state["rag_model_name"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


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

        if st.button("Clear chat & knowledge base"):
            st.session_state["vectorstore"] = None
            st.session_state["rag_model_name"] = None
            st.session_state["chat_history"] = []
            st.session_state["last_sources"] = []
            st.success("Cleared chat history and knowledge base.")

        if st.button("Build knowledge base"):
            with st.spinner("Processing documents and building vector store..."):
                docs = build_documents(uploaded_files or [], website_url.strip() or None)
                vectorstore = build_vectorstore(docs)

                if vectorstore is None:
                    st.error("No valid content found. Please upload files or provide a URL.")
                else:
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["rag_model_name"] = "openai/gpt-4.1-mini"
                    st.success(
                        f"Knowledge base built from {len(docs)} document chunks. You can now start chatting."
                    )

    col_chat, col_sources = st.columns([2, 1])

    with col_chat:
        st.subheader("Chat")

        if st.session_state["vectorstore"] is None:
            st.info("Build the knowledge base in the sidebar to start chatting.")
        else:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask a question about your data...")
            if user_input:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        vectorstore: FAISS = st.session_state["vectorstore"]
                        model_name = st.session_state.get("rag_model_name") or model_name
                        answer, sources = run_rag_query(
                            vectorstore=vectorstore,
                            model_name=model_name,
                            query=user_input,
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
