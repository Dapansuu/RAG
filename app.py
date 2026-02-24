from urllib import response

import streamlit as st
import os
import tempfile
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16192b 100%);
        border-right: 1px solid #2d3148;
    }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #2d3f7b, #3a4f9a);
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #fff;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(58,79,154,0.3);
    }
    .bot-msg {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #2d3148;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e0e4ff;
        max-width: 85%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .source-badge {
        display: inline-block;
        background: #2d3f7b;
        color: #8ca0ff;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75em;
        margin: 2px 3px;
        border: 1px solid #3a4f9a;
    }
    .status-ready {
        background: #1a3a2a;
        border: 1px solid #2d6a4f;
        border-radius: 8px;
        padding: 8px 14px;
        color: #52b788;
        font-size: 0.9em;
    }
    .status-empty {
        background: #2a1a1a;
        border: 1px solid #6a2d2d;
        border-radius: 8px;
        padding: 8px 14px;
        color: #e07070;
        font-size: 0.9em;
    }
    /* Title */
    .main-title {
        font-size: 2.4em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #6b7db3; font-size: 1em; margin-top: 0; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    /* Input */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1a1d2e;
        border: 1px solid #2d3148;
        color: #e0e4ff;
        border-radius: 8px;
    }
    .stSelectbox > div > div {
        background: #1a1d2e;
        border: 1px solid #2d3148;
        color: #e0e4ff;
    }
    hr { border-color: #2d3148; }
    .chunk-info {
        font-size: 0.75em;
        color: #6b7db3;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #2d3148;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Load LangChain components lazily ────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )



def get_llm(api_key: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.1,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://rag-chatbot.app",
            "X-Title": "RAG Chatbot",
        },
    )

# ─── Document Loaders ────────────────────────────────────────────────────────
def load_pdf(file_path: str):
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(file_path).load()


def load_txt(file_path: str):
    from langchain_community.document_loaders import TextLoader
    return TextLoader(file_path, encoding="utf-8").load()


def load_csv(file_path: str):
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(file_path).load()

def load_text_input(text: str):
    from langchain.docstore.document import Document
    return [Document(page_content=text, metadata={"source": "manual_input"})]



from langchain_core.documents import Document
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        for tag in soup(["script", "Style"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n")
        clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
        return [Document(page_content=clean_text)]
    
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return []
# ─── Build Vector Store ───────────────────────────────────────────────────────
def build_vectorstore(docs, chunk_size: int, chunk_overlap: int):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


# ─── Build RAG Chain ──────────────────────────────────────────────────────────
def build_rag_chain(vectorstore, llm, k: int, chain_type: str):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type=chain_type,
        verbose=False,
    )
    return chain


# ─── Session State ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "rag_chain": None,
        "doc_count": 0,
        "chunk_count": 0,
        "data_source_label": "",
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📚 Data Source")

    source_type = st.radio(
        "Source Type",
        ["📄 PDF", "📝 Text File", "📊 CSV File", "🌐 Website URL", "✏️ Paste Text"],
        label_visibility="collapsed",
    )

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 256, 2000, 800, 64,
                               help="Size of text chunks in tokens")
        chunk_overlap = st.slider("Chunk Overlap", 0, 400, 100, 16,
                                  help="Overlap between consecutive chunks")
        top_k = st.slider("Top-K Retrieval", 1, 10, 4,
                          help="Number of chunks retrieved per query")
        chain_type = st.selectbox("Chain Type", ["stuff", "map_reduce", "refine"],
                                  help="How context chunks are combined")

    st.markdown("---")

    # Source input UI
    uploaded_file = None
    url_input = ""
    text_input = ""

    if source_type in ["📄 PDF", "📝 Text File", "📊 CSV File"]:
        ext_map = {"📄 PDF": ["pdf"], "📝 Text File": ["txt", "md"], "📊 CSV File": ["csv"]}
        uploaded_file = st.file_uploader(
            "Upload File",
            type=ext_map[source_type],
            label_visibility="collapsed",
        )
    elif source_type == "🌐 Website URL":
        url_input = st.text_input("Website URL", placeholder="https://example.com/page",
                                  label_visibility="collapsed")
    else:  # Paste Text
        text_input = st.text_area("Paste your text here...", height=200,
                                  label_visibility="collapsed",
                                  placeholder="Paste any text content here and the chatbot will answer questions about it.")

    # Process button
    can_process = bool((...))

    if st.button("🚀 Build Knowledge Base", disabled=not can_process, use_container_width=True):
        st.session_state.processing = True
        with st.spinner("Processing your data source..."):
            try:
                docs = []
                label = ""

                if uploaded_file:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    if suffix == ".pdf":
                        docs = load_pdf(tmp_path)
                    elif suffix == ".csv":
                        docs = load_csv(tmp_path)
                    else:
                        docs = load_txt(tmp_path)
                    os.unlink(tmp_path)
                    label = uploaded_file.name

                elif url_input:
                    docs = scrape_website(url_input)
                    label = url_input

                else:
                    docs = load_text_input(text_input)
                    label = "Pasted Text"

                if not docs:
                    st.error("No content could be extracted from the source.")
                else:
                    vectorstore, chunk_count = build_vectorstore(docs, chunk_size, chunk_overlap)
                    llm = get_llm(api_key)
                    rag_chain = build_rag_chain(vectorstore, llm, top_k, chain_type)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.rag_chain = rag_chain
                    st.session_state.doc_count = len(docs)
                    st.session_state.chunk_count = chunk_count
                    st.session_state.data_source_label = label
                    st.session_state.messages = []  # reset chat
                    st.success(f"✅ Ready! {len(docs)} pages → {chunk_count} chunks indexed.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.session_state.processing = False

    # Status indicator
    st.markdown("---")
    if st.session_state.vectorstore:
        st.markdown(f"""<div class="status-ready">
            ✅ <b>Knowledge Base Ready</b><br>
            📄 Source: {st.session_state.data_source_label[:40]}{'...' if len(st.session_state.data_source_label)>40 else ''}<br>
            📦 {st.session_state.chunk_count} chunks indexed
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="status-empty">
            ⚠️ No knowledge base loaded.<br>
            Configure a data source and click Build.
        </div>""", unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                # reset memory
                st.session_state.rag_chain.memory.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="color:#4a5280; font-size:0.8em; text-align:center;">
    RAG Chatbot • Powered by<br>LangChain + OpenRouter<br>
    Embeddings: all-MiniLM-L6-v2
</div>""", unsafe_allow_html=True)


# ─── Main Area ────────────────────────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-title"> RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about your documents, websites, or custom text.</p>',
                unsafe_allow_html=True)

st.markdown("---")

# Welcome message
if not st.session_state.messages:
    with st.container():
        st.markdown("""
<div style="text-align:center; padding: 40px 20px; color: #4a5280;">
    <div style="font-size: 3em;">📖</div>
    <h3 style="color: #667eea; margin: 10px 0;">Get Started</h3>
    1. Choose a <b style="color:#8ca0ff">data source</b> (PDF, URL, text, etc.)<br>
    2. Click <b style="color:#8ca0ff">Build Knowledge Base</b><br>
    3. Start asking questions! 🚀</p>
""", unsafe_allow_html=True)

# Chat history display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            badges = "".join(
                f'<span class="source-badge">📄 {s}</span>'
                for s in msg["sources"]
            )
            sources_html = f'<div class="chunk-info">Sources: {badges}</div>'

        st.markdown(
            f'<div class="bot-msg">🤖 {msg["content"]}{sources_html}</div>',
            unsafe_allow_html=True,
        )

# ─── Chat Input ───────────────────────────────────────────────────────────────
st.markdown("")
with st.form(key="chat_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_query = st.text_input(
            "Ask a question",
            placeholder="Ask anything about your data source...",
            label_visibility="collapsed",
            disabled=st.session_state.vectorstore is None,
        )
    with col_btn:
        submit = st.form_submit_button(
            "Send ➤",
            use_container_width=True,
            disabled=st.session_state.vectorstore is None,
        )

if submit and user_query.strip():
    if not st.session_state.rag_chain:
        st.warning("Please build a knowledge base first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain({"question": user_query})
                answer = result.get("answer", "I couldn't find a relevant answer.")

                # Extract source labels
                source_docs = result.get("source_documents", [])
                sources = []
                seen = set()
                for doc in source_docs:
                    src = doc.metadata.get("source", "")
                    page = doc.metadata.get("page", "")
                    label = Path(src).name if src else "source"
                    if page != "":
                        label = f"{label} p.{int(page)+1}"
                    if label not in seen:
                        seen.add(label)
                        sources.append(label)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources[:4],  # show max 4 source badges
                })

            except Exception as e:
                err_msg = str(e)
                if "401" in err_msg:
                    err_msg = "Invalid API key. Please check your OpenRouter API key."
                elif "429" in err_msg:
                    err_msg = "Rate limit reached. Please wait a moment and try again."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {err_msg}",
                    "sources": [],
                })

        st.rerun()

# Footer hint
if st.session_state.vectorstore and not st.session_state.messages:
    st.markdown("""
<div style="text-align:center; color:#4a5280; font-size:0.85em; padding:20px;">
    ✅ Knowledge base is ready! Type a question above to get started.
</div>""", unsafe_allow_html=True)