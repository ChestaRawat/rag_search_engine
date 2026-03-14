# rag_pipeline.py
# THE BRAIN of the project
# Connects everything: FAISS + Tavily + Groq LLM
# Runs the full RAG pipeline and returns answer with citations

import os
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from models import RAGResponse, AnswerSource
from vector_store import search_documents, load_faiss_index, index_exists
from web_search import search_web, format_web_results
from query_router import classify_query, get_emoji


# ── Prompt Templates ──

DOC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.
Answer ONLY using the document context below.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."
Always mention which document you used.

Document Context:
{context}

Question: {question}

Answer with citations like [Doc: filename - Chunk N]:"""
)

WEB_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant with real-time web search.
Answer using the web search results below.
Always cite the web sources.

Web Search Results:
{context}

Question: {question}

Answer with citations like [Web: website name]:"""
)

HYBRID_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.
Answer using BOTH document context and web results below.
Clearly mention which info came from documents and which from web.

Context:
{context}

Question: {question}

Answer with citations [Doc: filename] and [Web: source]:"""
)


def get_llm():
    # Returns Groq LLM (FREE)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found! Get free key at console.groq.com")

    return ChatGroq(
        model="llama-3.3-70b-versatile",   # free Llama 3 model
        temperature=0.2,           # lower = more factual
        max_tokens=1000,
        api_key=api_key
    )


def build_doc_context(doc_results: List[dict]) -> str:
    # Formats document chunks into context string for LLM
    if not doc_results:
        return "No document results found."

    parts = []
    for i, r in enumerate(doc_results[:4]):
        parts.append(
            f"[Doc {i+1}] {r['title']} - Chunk {r['chunk_index']}\n"
            f"Content: {r['content']}"
        )
    return "\n\n".join(parts)


def build_hybrid_context(doc_results, web_results) -> str:
    # Combines doc and web results into one context string
    parts = []
    if doc_results:
        parts.append("=== FROM YOUR DOCUMENTS ===")
        parts.append(build_doc_context(doc_results))
    if web_results:
        parts.append("\n=== FROM WEB SEARCH ===")
        parts.append(format_web_results(web_results))
    return "\n\n".join(parts)


def run_rag_pipeline(question: str, db=None, use_web_search: bool = True) -> RAGResponse:
    # MAIN FUNCTION - runs the complete pipeline
    # 1. Classify query
    # 2. Retrieve from FAISS and/or Tavily
    # 3. Build context
    # 4. Generate answer with Groq
    # 5. Return response with citations

    print(f"\nQuestion: {question}")

    # Step 1: Classify query
    query_type, reason = classify_query(question)
    print(f"Query type: {query_type} | Reason: {reason}")

    # Force document if web search is disabled
    if not use_web_search and query_type in ["web", "hybrid"]:
        query_type = "document"

    # Step 2a: Search FAISS (for document and hybrid queries)
    doc_results = []
    if query_type in ["document", "hybrid"]:
        if db is not None:
            doc_results = search_documents(question, db)
        elif index_exists():
            db = load_faiss_index()
            doc_results = search_documents(question, db)

    # Step 2b: Search web (for web and hybrid queries)
    web_results = []
    if query_type in ["web", "hybrid"] and use_web_search:
        try:
            web_results = search_web(question)
        except Exception as e:
            print(f"Web search failed: {e}")
            # fallback to document search
            if not doc_results and db is not None:
                doc_results = search_documents(question, db)
            query_type = "document"

    # Step 3: Build context
    if query_type == "document":
        context = build_doc_context(doc_results)
        prompt = DOC_PROMPT
    elif query_type == "web":
        context = format_web_results(web_results)
        prompt = WEB_PROMPT
    else:  # hybrid
        context = build_hybrid_context(doc_results, web_results)
        prompt = HYBRID_PROMPT

    # Step 4: Generate answer with Groq
    try:
        llm = get_llm()
        formatted_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # Step 5: Build sources for citation display
    sources = []

    for r in doc_results:
        sources.append(AnswerSource(
            source_type="document",
            title=r['title'],
            content=r['content'],
            citation=r['citation'],
            url=None
        ))

    for r in web_results:
        sources.append(AnswerSource(
            source_type="web",
            title=r.title,
            content=r.content,
            citation=f"[Web] {r.title}",
            url=r.url
        ))

    return RAGResponse(
        question=question,
        answer=answer,
        sources=sources,
        query_type=query_type,
        doc_chunks=doc_results,
        web_results=web_results
    )