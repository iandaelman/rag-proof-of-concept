# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from app.vector_store import retrieve_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query = "Vivian VACHAUDEZ"
    vector_store = retrieve_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    relevant_docs = retriever.invoke(query)
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
