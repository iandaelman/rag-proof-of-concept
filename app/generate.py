from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


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
    model = ChatOllama(model=model_name)
    messages = [
        SystemMessage(content="You are an expert in answering accurate questions about the IT support process of MyMinfin. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)
