import streamlit as st
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Azure Connection
conn_str = "DefaultEndpointsProtocol=https;AccountName=foraiembedding;AccountKey=wT6d+noyeQd0g/A055uWFPSBz0ZJMylLQMdqpX0qOUYIvn5wQ9g6DXLsPw1XCyoVyxI4reVyxDDw+AStaD3V6w==;EndpointSuffix=core.windows.net"
container_name = "doc"
# Connect to Azure Blob
blob_service = BlobServiceClient.from_connection_string(conn_str)
container_client = blob_service.get_container_client(container_name)
documents = []
# Read documents
for blob in container_client.list_blobs():
    blob_client = container_client.get_blob_client(blob.name)
    content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
    documents.append(content)
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Create embeddings
doc_embeddings = model.encode(documents)
# Streamlit UI
st.title("Azure AI Semantic Search Demo")
st.write("Ask questions from Azure Blob documents")
# User input
query = st.text_input("Ask your question")
if query:
    # Query embedding
    query_embedding = model.encode([query])
    # Similarity check
    scores = cosine_similarity(query_embedding, doc_embeddings)
    # Best match
    best_index = np.argmax(scores)
    st.subheader("Most Relevant Answer")
    st.write(documents[best_index])