
# query_router.py
# Decides WHERE to search based on the question
# Returns: "document", "web", or "hybrid"

# Words that suggest user wants LIVE/CURRENT info
WEB_KEYWORDS = [
    "latest", "current", "today", "recent", "now",
    "2024", "2025", "2026", "this week", "this month",
    "news", "update", "new", "trending", "breaking",
    "live", "real-time", "price", "stock", "happening","web"
]

# Words that suggest user wants CONCEPTUAL info from docs
DOC_KEYWORDS = [
    "explain", "what is", "define", "how does", "describe",
    "tell me about", "overview", "example", "tutorial",
    "guide", "learn", "understand", "concept", "theory",
    "difference between", "algorithm", "method", "steps"
]


def classify_query(query: str):
    # Returns (query_type, reason)
    query_lower = query.lower()

    web_matches = [kw for kw in WEB_KEYWORDS if kw in query_lower]
    doc_matches = [kw for kw in DOC_KEYWORDS if kw in query_lower]

    if web_matches and doc_matches:
        return "hybrid", f"Mixed query: found {web_matches} and {doc_matches}"
    elif web_matches:
        return "web", f"Time-sensitive keywords: {web_matches}"
    elif doc_matches:
        return "document", f"Conceptual keywords: {doc_matches}"
    else:
        return "document", "No specific keywords - defaulting to document search"


def get_emoji(query_type: str) -> str:
    return {"document": "📄", "web": "🌐", "hybrid": "🔀"}.get(query_type, "❓")


def get_label(query_type: str) -> str:
    return {"document": "Document Search", "web": "Web Search", "hybrid": "Hybrid Search"}.get(query_type, "Unknown")
