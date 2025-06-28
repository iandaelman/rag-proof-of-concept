import os

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

from app import vector_store
from app.utils.RetrievalMethod import RetrievalMethod
from app.utils.State import State
from app.vector_store import init_vector_store


def retrieve(state: State):
    print("\n--- Retrieving documents ---")
    retrieved_docs = query_vector_store(state["persistent_directory"],
                                        state["store_name"],
                                        state["query"],
                                        state["retrieval_method"].value,
                                        state["search_kwargs"])
    print("\n--- Retrieved documents ---")
    return {"context": retrieved_docs}


def query_vector_store(persistent_directory, store_name, query, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        return relevant_docs
    else:
        print(f"Vector store {store_name} does not exist. Therefor we could not return any documents")
        return None


def test_different_retrieval_methods(store_name="chroma_db",
                                     embeddings=HuggingFaceEmbeddings(
                                         model_name="sentence-transformers/all-MiniLM-L6-v2"), query=None):
    # Showcase different retrieval methods

    # 1. Similarity Search
    # This method retrieves documents based on vector similarity.
    # It finds the most similar documents to the query vector based on cosine similarity.
    # Use this when you want to retrieve the top k most similar documents.
    print("\n--- Using Similarity Search ---")
    query_vector_store(store_name, query,
                       embeddings, "similarity", {"k": 3})

    # 2. Max Marginal Relevance (MMR)
    # This method balances between selecting documents that are relevant to the query and diverse among themselves.
    # 'fetch_k' specifies the number of documents to initially fetch based on similarity.
    # 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
    # Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
    # Note: Relevance measures how closely documents match the query.
    # Note: Diversity ensures that the retrieved documents are not too similar to each other,
    #       providing a broader range of information.
    print("\n--- Using Max Marginal Relevance (MMR) ---")
    query_vector_store(
        store_name,
        query,
        embeddings,
        "mmr",
        {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
    )

    # 3. Similarity Score Threshold
    # This method retrieves documents that exceed a certain similarity score threshold.
    # 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
    # Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
    print("\n--- Using Similarity Score Threshold ---")
    query_vector_store(
        store_name,
        query,
        embeddings,
        "similarity_score_threshold",
        {"score_threshold": 0.1},
    )

    print("Querying demonstrations with different search types completed.")


@tool(response_format="content_and_artifact")
def retrieve_tool_chain(query: str):
    """Retrieve information related to a query."""
    store_name = "chroma_db_doc"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    persistent_directory = os.path.join(base_dir, "vector_db", store_name)
    vector_store = init_vector_store(document_path="knowledge-base-doc", db_name=store_name)

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    llm = ChatOllama(temperature=0.1, model="llama3.2")
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve_tool_chain])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}
