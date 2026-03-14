# rag_search_engine

🔍 Hybrid RAG Search Engine
Python LangChain Streamlit FAISS Groq Tavily

A smart AI-powered search engine that searches your documents AND the live web — with citations.

🚀 Quick Start · 🧠 How It Works · 📁 Project Structure · 🛠️ Tech Stack · 🐛 Troubleshooting

✨ What Does This Do?
Upload your PDFs → Ask any question → Get a cited answer from your documents and/or the live web.

You:  "Explain machine learning and what are the latest AI tools in 2025?"
         ↓
  🔀 Hybrid Search detected
         ↓
  📄 Searches your PDFs   +   🌐 Searches the web (Tavily)
         ↓
  🤖 Groq (Llama3) generates answer with citations
         ↓
Bot:  "Machine learning is... [Doc: AI.pdf - Chunk 4]
       Recent tools include... [Web: TechCrunch]"
🚀 Quick Start
Step 1 — Clone the repo
git clone https://github.com/yourusername/rag-search-engine.git
cd rag-search-engine
Step 2 — Create virtual environment
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
Step 3 — Install dependencies
pip install -r requirements.txt
Step 4 — Get your FREE API keys
Service	Link	Cost
Groq (LLM)	console.groq.com	🆓 Free
Tavily (Web Search)	tavily.com	🆓 Free tier
Step 5 — Add API keys to .env
GROQ_API_KEY=gsk_your-key-here
TAVILY_API_KEY=tvly-your-key-here
Step 6 — Run!
streamlit run app.py
Open browser at → http://localhost:8501 🎉

🧠 How It Works
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD FLOW                      │
│                                                             │
│  PDF Upload → LangChain reads → Text cleaned → Split into   │
│  500-char chunks → HuggingFace converts to vectors →        │
│  FAISS stores vectors on disk                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUESTION ANSWER FLOW                      │
│                                                             │
│  Question → Query Router classifies                         │
│      ↓              ↓               ↓                       │
│   📄 Doc          🌐 Web         🔀 Hybrid                  │
│   FAISS          Tavily        FAISS + Tavily               │
│      └──────────────┴───────────────┘                       │
│                       ↓                                     │
│              Context assembled                              │
│                       ↓                                     │
│           Groq Llama3 generates answer                      │
│                       ↓                                     │
│          Answer + Citations shown in UI                     │
└─────────────────────────────────────────────────────────────┘
🔀 Query Routing Logic
Question Type	Example	Routed To
Conceptual	"What is machine learning?"	📄 Documents
Current events	"Latest AI news today?"	🌐 Web
Mixed	"Explain AI and recent 2025 tools?"	🔀 Hybrid
📁 Project Structure
rag_project/
│
├── 📄 app.py              ← Streamlit UI (run this!)
├── 📄 models.py           ← Data blueprints
├── 📄 ingestion.py        ← Reads PDFs, text, Wikipedia
├── 📄 chunking.py         ← Splits docs into chunks
├── 📄 vector_store.py     ← FAISS vector database
├── 📄 web_search.py       ← Tavily web search
├── 📄 query_router.py     ← Decides doc/web/hybrid
├── 📄 rag_pipeline.py     ← Main brain (connects all)
│
├── 🔑 .env                ← Your API keys (never share!)
├── 📦 requirements.txt    ← Python dependencies
│
├── 📁 docs/               ← Put your PDFs here
└── 📁 faiss_index/        ← Auto-created after indexing
🛠️ Tech Stack
Technology	Role	Cost
LangChain	AI orchestration framework (the glue)	Free
FAISS	Vector database by Facebook	Free
HuggingFace	Free embedding model (text → vectors)	Free
Tavily	Real-time web search API	Free tier
Groq + Llama3	LLM for answer generation	Free
Streamlit	Web UI framework	Free
PyPDF	PDF reading	Free
💡 This entire project runs for FREE using Groq and Tavily free tiers!

💬 Example Questions
📄 Document Questions (searches your PDFs)
"What is the attention mechanism in transformers?"
"Explain RAG in simple terms"
"Define machine learning"
"How does FAISS work?"
🌐 Web Questions (searches internet live)
"Latest AI developments today"
"Current OpenAI models in 2025"
"Recent news about LLMs this week"
"What AI tools were released recently?"
🔀 Hybrid Questions (searches both!)
"Explain AI and what are the latest tools in 2025?"
"What is machine learning and recent advances?"
"Describe RAG and current state of the art systems?"
🖥️ UI Features
📄 Upload multiple PDFs — index any documents
🌍 Wikipedia loader — load any Wikipedia article
🔀 Automatic routing — smart doc vs web decision
📊 Source tabs — see exactly where each answer came from
🗂️ Evidence expanders — read the actual chunks used
🔗 Web citations — clickable links to web sources
🌐 Toggle web search — enable/disable Tavily anytime
🗑️ Clear index — reset and start fresh
🐛 Troubleshooting
ModuleNotFoundError: No module named 'langchain'
GROQ_API_KEY not found
faiss-cpu installation fails on Windows
Model decommissioned error from Groq
Port already in use
🔑 Environment Variables
Variable	Description	Where to get
GROQ_API_KEY	Groq LLM API key	console.groq.com
TAVILY_API_KEY	Tavily web search key	tavily.com
🤖 Supported Groq Models
Model	Speed	Quality
llama-3.3-70b-versatile	Medium	⭐⭐⭐ Best
llama-3.1-8b-instant	Fast	⭐⭐ Good
gemma2-9b-it	Fast	⭐⭐ Good
mixtral-8x7b-32768	Medium	⭐⭐⭐ Good
Change the model in rag_pipeline.py → get_llm() function.

📚 Key Concepts Learned
By building this project you understand:

✅ RAG — Retrieval Augmented Generation
✅ Vector embeddings — converting text to numbers
✅ Semantic search — search by meaning not keywords
✅ FAISS — vector database for fast similarity search
✅ LangChain — AI application framework
✅ Hybrid retrieval — combining vector + web search
✅ Citation-aware generation — grounded AI answers
✅ Streamlit — building AI web apps with Python
Built with ❤️ using LangChain + FAISS + Groq + Tavily + Streamlit

⭐ Star this repo if it helped you!
