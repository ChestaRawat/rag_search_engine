# vector_store.py
# Manages the FAISS vector database
# Stores document chunks as vectors and searches them

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from models import DocumentChunk
from chunking import chunks_to_langchain_docs


# Settings
FAISS_INDEX_PATH = "faiss_index"          # folder to save index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free model
TOP_K_RESULTS = 4                          # how many chunks to return


def get_embeddings():
    # Loads the HuggingFace embedding model
    # Downloads ~90MB first time, then works offline
    print("Loading HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding model loaded!")
    return embeddings


def index_documents(chunks: List[DocumentChunk]) -> FAISS:
    # Creates or updates FAISS index with new chunks
    print(f"Indexing {len(chunks)} chunks into FAISS...")

    embeddings = get_embeddings()
    lc_docs = chunks_to_langchain_docs(chunks)  # convert to LangChain format

    if os.path.exists(FAISS_INDEX_PATH):
        # Add to existing index
        print("Adding to existing FAISS index...")
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_documents(lc_docs)
    else:
        # Create brand new index
        print("Creating new FAISS index...")
        db = FAISS.from_documents(documents=lc_docs, embedding=embeddings)

    # Save to disk so it persists after app restart
    db.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved!")
    return db


def load_faiss_index() -> FAISS:
    # Loads saved FAISS index from disk
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Please upload documents first.")

    print("Loading FAISS index from disk...")
    embeddings = get_embeddings()
    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded!")
    return db


def search_documents(query: str, db: FAISS, k: int = TOP_K_RESULTS) -> List[dict]:
    # Searches FAISS for most relevant chunks
    # Returns top K most similar chunks to the query
    print(f"Searching FAISS for: '{query}'")

    results_with_scores = db.similarity_search_with_score(query, k=k)

    formatted_results = []
    for lc_doc, score in results_with_scores:
        result = {
            "content": lc_doc.page_content,
            "title": lc_doc.metadata.get("title", "Unknown"),
            "source_type": lc_doc.metadata.get("source_type", "unknown"),
            "chunk_index": lc_doc.metadata.get("chunk_index", 0),
            "chunk_id": lc_doc.metadata.get("chunk_id", ""),
            "score": float(score),
            "citation": f"[Doc] {lc_doc.metadata.get('title', 'Unknown')} - Chunk {lc_doc.metadata.get('chunk_index', 0)}"
        }
        formatted_results.append(result)

    print(f"Found {len(formatted_results)} relevant chunks")
    return formatted_results


def index_exists() -> bool:
    # Returns True if FAISS index already exists on disk
    return os.path.exists(FAISS_INDEX_PATH)


def clear_index():
    # Deletes the FAISS index completely
    import shutil
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print("FAISS index cleared!")