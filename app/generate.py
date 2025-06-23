from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.utils.State import State


def generate(state: State):
    print("\n--- Generating awnser ---")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", """Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Question: {query} 
        Context: {context}
        Please provide an answer based only on the provided context. If the answer is not found in the context, respond with 'I'm not sure'.""")])
    print("\n--- Initializing LLM model ---")
    model = ChatOllama(temperature=0.1, model=state["model_name"])
    messages = prompt_template.invoke({"query": state["query"], "context": docs_content})
    print(f"\n--- Invoking LLM model with {messages} ---")
    response = model.invoke(messages)
    return {"response": response}
