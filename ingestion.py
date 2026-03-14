# ingestion.py
# This file READS documents (PDFs, text files, Wikipedia)
# Uses LangChain loaders to do the heavy lifting

import os
import re
import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader

from models import Document


def clean_text(text: str) -> str:
    # Removes extra spaces, weird characters from text
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)           # remove extra spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove weird characters
    text = text.strip()
    return text


def load_pdf(file_path: str) -> Document:
    # Reads a PDF file and returns a Document object
    print(f"Loading PDF: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()  # LangChain reads each page

    # Combine all pages into one string
    full_content = ""
    for page in pages:
        full_content += page.page_content + "\n"

    cleaned = clean_text(full_content)
    title = os.path.basename(file_path)

    return Document(
        source_id=str(uuid.uuid4()),
        source_type="pdf",
        title=title,
        content=cleaned,
        metadata={"file_path": file_path, "num_pages": len(pages)}
    )


def load_text_file(file_path: str) -> Document:
    # Reads a .txt or .md file
    print(f"Loading text file: {file_path}")

    loader = TextLoader(file_path, encoding='utf-8')
    pages = loader.load()

    full_content = "\n".join([p.page_content for p in pages])
    cleaned = clean_text(full_content)
    title = os.path.basename(file_path)

    return Document(
        source_id=str(uuid.uuid4()),
        source_type="text",
        title=title,
        content=cleaned,
        metadata={"file_path": file_path}
    )


def load_wikipedia(topic: str) -> Document:
    # Fetches a Wikipedia article
    print(f"Loading Wikipedia: {topic}")

    loader = WikipediaLoader(query=topic, load_max_docs=1, doc_content_chars_max=5000)
    pages = loader.load()

    if not pages:
        raise ValueError(f"No Wikipedia page found for: {topic}")

    cleaned = clean_text(pages[0].page_content)
    title = pages[0].metadata.get('title', topic)

    return Document(
        source_id=str(uuid.uuid4()),
        source_type="wikipedia",
        title=title,
        content=cleaned,
        metadata={"wiki_url": pages[0].metadata.get('source', '')}
    )


def load_uploaded_file(file_path: str) -> Document:
    # Auto detects file type and loads it
    # Used when user uploads file in Streamlit
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext in ['.txt', '.md']:
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")