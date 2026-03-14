# models.py
# These are just blueprints/templates for data
# Like creating a form with fixed fields

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Document:
    # Represents a full document (PDF, text, wikipedia)
    source_id: str        # unique ID like "doc_001"
    source_type: str      # "pdf", "wikipedia", "text"
    title: str            # document name
    content: str          # full text
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    # Represents a small piece of a document
    chunk_id: str         # unique chunk ID
    source_id: str        # which document it came from
    source_type: str      # "pdf", "wikipedia", "text"
    title: str            # document title
    content: str          # chunk text
    chunk_index: int      # which chunk number
    metadata: dict = field(default_factory=dict)


@dataclass
class WebSearchResult:
    # Represents one web search result from Tavily
    title: str            # webpage title
    content: str          # snippet text
    url: str              # webpage link
    score: float = 0.0    # relevance score


@dataclass
class AnswerSource:
    # Represents one source used in the final answer
    source_type: str      # "document" or "web"
    title: str            # source title
    content: str          # relevant content
    citation: str         # formatted citation string
    url: Optional[str] = None  # only for web sources


@dataclass
class RAGResponse:
    # The final response returned to Streamlit UI
    question: str         # original question
    answer: str           # generated answer
    sources: list         # list of AnswerSource
    query_type: str       # "document", "web", "hybrid"
    doc_chunks: list = field(default_factory=list)
    web_results: list = field(default_factory=list)