import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found.")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📚 RAG Q&A System")

# -----------------------------
# Clear Chat Button
# -----------------------------
if st.button("🗑 Clear Chat"):
    st.cache_resource.clear()
    st.session_state.messages = []
    st.success("Chat and vector database cleared.")
    st.rerun()

# -----------------------------
# Source Selection
# -----------------------------
source_type = st.selectbox(
    "Select Knowledge Source",
    ["PDF", "Text File", "Website"]
)

uploaded_file = None
website_url = None

if source_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

elif source_type == "Text File":
    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

elif source_type == "Website":
    website_url = st.text_input("Enter website URL")

# -----------------------------
# Process Document
# -----------------------------
@st.cache_resource
def process_document(file_path=None, file_type=None, url=None):

    # Load documents
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

    elif url:
        loader = WebBaseLoader(url)
        documents = loader.load()

    else:
        return None

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings (OpenRouter)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    # Create vector DB
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vectordb

# -----------------------------
# Handle Source Processing
# -----------------------------
vectordb = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    file_type = uploaded_file.name.split(".")[-1].lower()

    vectordb = process_document(
        file_path=tmp_path,
        file_type=file_type
    )

    st.success(f"{source_type} processed successfully!")

elif website_url:
    vectordb = process_document(url=website_url)
    st.success("Website processed successfully!")

# -----------------------------
# Initialize RAG Chain
# -----------------------------
if vectordb:

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -----------------------------
    # Chat History
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask something about the document..."):

        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )