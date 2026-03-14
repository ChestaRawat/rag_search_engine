# chunking.py
# Splits long documents into smaller pieces called "chunks"
# Because AI models cant process very long text at once

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDoc
from models import Document, DocumentChunk


# Settings
CHUNK_SIZE = 500     # each chunk = 500 characters
CHUNK_OVERLAP = 50   # 50 characters shared between chunks


def chunk_document(document: Document) -> List[DocumentChunk]:
    # Splits one document into many chunks

    # LangChain splitter - tries to split on paragraphs first,
    # then sentences, then words, then characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split the text
    text_chunks = splitter.split_text(document.content)

    # Convert each piece to DocumentChunk
    document_chunks = []
    for index, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue  # skip empty chunks

        chunk = DocumentChunk(
            chunk_id=f"{document.source_id}_chunk_{index}",
            source_id=document.source_id,
            source_type=document.source_type,
            title=document.title,
            content=chunk_text,
            chunk_index=index,
            metadata={
                "chunk_index": index,
                "source_title": document.title,
                "source_type": document.source_type
            }
        )
        document_chunks.append(chunk)

    print(f"Chunked '{document.title}': {len(document_chunks)} chunks")
    return document_chunks


def chunk_multiple_documents(documents: List[Document]) -> List[DocumentChunk]:
    # Chunks all documents in a list
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def chunks_to_langchain_docs(chunks: List[DocumentChunk]) -> List[LangChainDoc]:
    # Converts our chunks to LangChain format
    # Needed because FAISS expects LangChain Document objects
    langchain_docs = []
    for chunk in chunks:
        lc_doc = LangChainDoc(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
        )
        langchain_docs.append(lc_doc)
    return langchain_docs