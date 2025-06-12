import os

import numpy as np
import plotly.graph_objects as go
from langchain_chroma import Chroma
from sklearn.manifold import TSNE
from langchain_huggingface import HuggingFaceEmbeddings




def create_datastore(documents, db_name = "chroma_knowledge_base_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch

    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore

def visualise_vector_store(vectorstore):
    # Direct access for visualization (non-production use)
    collection = vectorstore._collection
    # Prework

    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
    colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in
              doc_types]
    # We humans find it easier to visalize things in 2D!
    # Reduce the dimensionality of the vectors to 2D using t-SNE
    # (t-distributed stochastic neighbor embedding)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    fig.show()