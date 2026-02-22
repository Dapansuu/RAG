import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain 1.x imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📚 RAG Q&A System")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# -----------------------------
# Process Document (Cached)
# -----------------------------
@st.cache_resource
def process_document(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings via OpenRouter
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    vectordb = process_document(tmp_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    st.success("Document processed and indexed successfully!")

    # -----------------------------
    # Initialize LLM
    # -----------------------------
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    # -----------------------------
    # Prompt Template
    # -----------------------------
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

    # -----------------------------
    # LCEL RAG Chain
    # -----------------------------
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -----------------------------
    # Question Input
    # -----------------------------
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(query)

        st.subheader("Answer")
        st.write(response)