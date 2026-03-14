🔍 Hybrid RAG Search Engine

A smart AI-powered search engine that retrieves information from your documents and the live web and generates accurate answers with citations.

Built using LangChain, FAISS, Groq LLMs, Tavily Web Search, and Streamlit.

🚀 Features

✅ Upload and search multiple PDFs
✅ Semantic search using vector embeddings
✅ Live web search for recent information
✅ Hybrid retrieval (documents + internet)
✅ Citation-aware answers
✅ Interactive Streamlit UI
✅ Expandable evidence view for transparency

🧠 How It Works

The system follows a Retrieval Augmented Generation (RAG) pipeline.

1️⃣ Document Ingestion
PDF Upload
     ↓
LangChain loads documents
     ↓
Text cleaning
     ↓
Chunking (500 characters)
     ↓
HuggingFace Embeddings
     ↓
FAISS Vector Database

Documents are converted into vector embeddings and stored in FAISS for semantic search.

2️⃣ Question Answering Flow
User Question
      ↓
Query Router
      ↓
┌───────────────┬───────────────┬───────────────┐
│ Document      │ Web Search    │ Hybrid        │
│ Retrieval     │ (Tavily)      │ (Both)        │
│ (FAISS)       │               │               │
└───────────────┴───────────────┴───────────────┘
                ↓
        Context Assembly
                ↓
        Groq Llama3 LLM
                ↓
       Final Answer + Citations



       
🔀 Query Routing Logic
Question Type	Example	Routed To
Conceptual	"What is machine learning?"	📄 Documents
Current Events	"Latest AI news today?"	🌐 Web
Mixed	"Explain AI and latest tools in 2025?"	🔀 Hybrid
📁 Project Structure
rag_project/
│
├── app.py              # Streamlit UI
├── models.py           # Data models
├── ingestion.py        # Document ingestion
├── chunking.py         # Text chunking logic
├── vector_store.py     # FAISS vector database
├── web_search.py       # Tavily web search
├── query_router.py     # Query classification
├── rag_pipeline.py     # Main RAG pipeline
│
├── requirements.txt
├── .env                # API keys
│
├── docs/               # Upload your PDFs
└── faiss_index/        # Generated vector index
⚙️ Tech Stack
Technology	Purpose
LangChain	AI orchestration framework
FAISS	Vector database for similarity search
HuggingFace	Text embeddings
Tavily	Real-time web search
Groq + Llama3	LLM for answer generation
Streamlit	Interactive web interface
PyPDF	PDF document parsing

💡 The project runs completely on free tiers using Groq and Tavily APIs.
