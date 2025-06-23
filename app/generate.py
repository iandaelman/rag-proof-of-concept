from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.utils.State import State


def generate_answer(relevant_documents=None, query=None, model_name="llama3.2"):
    print("\n--- Generating Answer with ---")
    print(f"\n--- Query: {query} ---")
    combined_input = (
            "Here are some documents that might help answer the question: "
            + query
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_documents])
            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    model = ChatOllama(temperature=0.1, model=model_name)
    messages = [
        SystemMessage(
            content="You are an expert in answering accurate questions about the IT support process of MyMinfin. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print("Full result:")
    print(result)
    print("Content only:")
    return {"answer": result.content}


def generate(state: State):
    print("\n--- Generating awnser ---")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", """Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Question: {query} 
        Context: {context} """)])
    print("\n--- Initializing LLM model ---")
    model = ChatOllama(temperature=0.1, model=state["model_name"])
    messages = prompt_template.invoke({"query": state["query"], "context": docs_content})
    print(f"\n--- Invoking LLM model with {messages} ---")
    response = model.invoke(messages)
    return {"response": response}
