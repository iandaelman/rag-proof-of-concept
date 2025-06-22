from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


def generate_answer(relevant_documents=None, query=None, model_name="llama3.2"):
    combined_input = (
            "Here are some documents that might help answer the question: "
            + query
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_documents])
            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    model = ChatOllama(model="gemma3:latest")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)
