from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.document_loader import load_documents_from_knowledge_base, split_documents
from app.embedder import create_datastore
from app.state import State


def initialize_llm(model="llama3.2"):
    llm = ChatOllama(model=model, temperature=7)
    return llm


# Define application steps
def retrieve(state: State):
    documents = load_documents_from_knowledge_base()
    chunks = split_documents(documents=documents)
    vector_store = create_datastore(chunks)
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Question: {question}

    Context: {context}

    Answer:
    """)
    llm = initialize_llm()
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    print(messages)
    response = llm.invoke(messages)
    print(response)
    return {"answer": response.content}