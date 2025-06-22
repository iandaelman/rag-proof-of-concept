import os

from enum import Enum
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RetrievalMethod(Enum):
    SIMILARITY_SEARCH = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"


def query_vector_store(persistent_directory, store_name, query, embedding_function, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
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


def retrieve_documents(
        query,
        store_name="chroma_db",
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        retrieval_method=RetrievalMethod.SIMILARITY_SEARCH):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", store_name)
    # Use the specified retrieval method or default to showcasing all methods
    if retrieval_method == RetrievalMethod.SIMILARITY_SEARCH:
        print("\n--- Using Similarity Search ---")
        return query_vector_store(persistent_directory,
                                  store_name,
                                  query,
                                  embeddings,
                                  retrieval_method.value,
                                  {"k": 3})

    elif retrieval_method == RetrievalMethod.MMR:
        print("\n--- Using Max Marginal Relevance (MMR) ---")
        return query_vector_store(
            persistent_directory,
            store_name,
            query,
            embeddings,
            retrieval_method.value,
            {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        )

    elif retrieval_method == RetrievalMethod.SIMILARITY_SCORE_THRESHOLD:
        print("\n--- Using Similarity Score Threshold ---")
        return query_vector_store(
            persistent_directory,
            store_name,
            query,
            embeddings,
            retrieval_method.value,
            {"score_threshold": 0.1},
        )

    else:
        # Default behavior: use Similarity Search
        print("\n--- Using Similarity Search ---")
        return query_vector_store(store_name, query, embeddings, "similarity", {"k": 3})


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
