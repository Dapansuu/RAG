from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say:
"I don't have enough information in the documents."

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {
        "context": retriever,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)
def ask_question(question: str) -> str:
    return rag_chain.invoke(question)

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answer = ask_question(q)
        print("\nAnswer:\n", answer)

    print("\nGoodbye!")