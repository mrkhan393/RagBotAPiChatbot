import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
vector_dim = model.get_sentence_embedding_dimension()

# Initialize FAISS index
index = faiss.IndexFlatL2(vector_dim)

# Store documents and metadata
documents = []  # Each entry: {"text": chunk_text, "metadata": {filename, page, chunk_index}}

def add_to_vectorstore(chunks, metadata=None):
    """
    Adds chunks of text with metadata to FAISS index
    """
    global documents
    if not chunks:
        return

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index.add(np.array(embeddings, dtype=np.float32))
    for i, chunk in enumerate(chunks):
        entry_metadata = metadata.copy() if metadata else {}
        entry_metadata["chunk_index"] = i
        documents.append({"text": chunk, "metadata": entry_metadata})

def search_vectorstore(query, top_k=5):
    """
    Searches FAISS index for similar text chunks
    """
    if index.ntotal == 0:
        return []

    q_embed = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_embed, dtype=np.float32), top_k)
    
    results = []
    for idx in I[0]:
        if 0 <= idx < len(documents):
            results.append(documents[idx])
    return results
