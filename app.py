# app.py
# The Streamlit UI - this is what you see in the browser
# Run with: streamlit run app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads keys from .env file automatically

from ingestion import load_uploaded_file, load_wikipedia
from chunking import chunk_document
from vector_store import index_documents, load_faiss_index, index_exists, clear_index
from rag_pipeline import run_rag_pipeline
from query_router import classify_query, get_emoji, get_label


# ── Page Setup ──
st.set_page_config(
    page_title="Hybrid RAG Search Engine",
    page_icon="🔍",
    layout="wide"
)


# ── Session State ──
# These variables persist across reruns of the app
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = True


# Load existing FAISS index if it exists on disk
if st.session_state.db is None and index_exists():
    try:
        st.session_state.db = load_faiss_index()
    except:
        pass


# ────────────────────────────
# SIDEBAR
# ────────────────────────────
with st.sidebar:
    st.title("📚 Document Manager")
    st.markdown("---")

    # Upload Documents
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Documents", type="primary"):
            all_chunks = []
            progress = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded_files):
                status.text(f"Processing {file.name}...")

                try:
                    # Save to temp file first
                    suffix = "." + file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name

                    # Load and chunk
                    doc = load_uploaded_file(tmp_path)
                    doc.title = file.name
                    chunks = chunk_document(doc)
                    all_chunks.extend(chunks)

                    if file.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(file.name)

                    os.unlink(tmp_path)  # delete temp file

                except Exception as e:
                    st.error(f"Error with {file.name}: {e}")

                progress.progress((i + 1) / len(uploaded_files))

            if all_chunks:
                status.text("Building FAISS index...")
                st.session_state.db = index_documents(all_chunks)
                st.success(f"✅ Indexed {len(uploaded_files)} file(s)!")

    st.markdown("---")

    # Wikipedia Loader
    st.subheader("🌍 Load Wikipedia")
    wiki_topic = st.text_input("Wikipedia Topic", placeholder="e.g. Machine Learning")

    if st.button("📖 Load Wikipedia"):
        if wiki_topic:
            with st.spinner(f"Loading {wiki_topic}..."):
                try:
                    doc = load_wikipedia(wiki_topic)
                    chunks = chunk_document(doc)
                    st.session_state.db = index_documents(chunks)
                    st.session_state.indexed_files.append(f"[Wiki] {doc.title}")
                    st.success(f"✅ Loaded: {doc.title}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a topic")

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    st.session_state.use_web_search = st.toggle(
        "🌐 Enable Web Search",
        value=st.session_state.use_web_search
    )

    st.markdown("---")

    # Indexed Files List
    st.subheader("📂 Indexed Files")
    if st.session_state.indexed_files:
        for f in st.session_state.indexed_files:
            st.markdown(f"✅ {f}")
    else:
        st.info("No files indexed yet")

    if st.button("🗑️ Clear All Documents"):
        clear_index()
        st.session_state.db = None
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.success("Cleared!")
        st.rerun()


# ────────────────────────────
# MAIN PANEL
# ────────────────────────────
st.title("🔍 Hybrid RAG Search Engine")
st.markdown("*Ask questions from your documents + the live web*")
st.markdown("---")

# Status Bar
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.db:
        st.success(f"📚 Knowledge Base Ready ({len(st.session_state.indexed_files)} files)")
    else:
        st.warning("📚 No documents uploaded yet")

with col2:
    if st.session_state.use_web_search and os.getenv("TAVILY_API_KEY"):
        st.success("🌐 Web Search: Active")
    elif st.session_state.use_web_search:
        st.warning("🌐 Web Search: Need Tavily key")
    else:
        st.info("🌐 Web Search: Disabled")

with col3:
    if os.getenv("GROQ_API_KEY"):
        st.success("🤖 Llama3 (Groq): Connected")
    else:
        st.error("🤖 Groq: Add API key in sidebar")

st.markdown("---")

# Display Chat History
for question, response in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        emoji = get_emoji(response.query_type)
        label = get_label(response.query_type)
        st.markdown(f"**{emoji} {label}**")
        st.write(response.answer)

        if response.sources:
            tab1, tab2, tab3 = st.tabs([
                "📝 Sources",
                f"📄 Documents ({len(response.doc_chunks)})",
                f"🌐 Web ({len(response.web_results)})"
            ])

            with tab1:
                for s in response.sources:
                    if s.source_type == "document":
                        st.markdown(f"📄 `{s.citation}`")
                    else:
                        st.markdown(f"🌐 `{s.citation}` — [link]({s.url})")

            with tab2:
                if response.doc_chunks:
                    for chunk in response.doc_chunks:
                        with st.expander(f"📄 {chunk['citation']}"):
                            st.write(chunk['content'])
                else:
                    st.info("No document chunks used")

            with tab3:
                if response.web_results:
                    for r in response.web_results:
                        with st.expander(f"🌐 {r.title}"):
                            st.write(r.content)
                            st.markdown(f"[🔗 Full article]({r.url})")
                else:
                    st.info("No web results used")


# Chat Input Box
question = st.chat_input(
    "Ask anything... (about your documents or anything on the web)",
    disabled=not os.getenv("GROQ_API_KEY")
)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            query_type, reason = classify_query(question)
            emoji = get_emoji(query_type)
            st.info(f"{emoji} Routing to **{query_type.upper()}** search — *{reason}*")

            try:
                response = run_rag_pipeline(
                    question=question,
                    db=st.session_state.db,
                    use_web_search=st.session_state.use_web_search
                )

                label = get_label(response.query_type)
                st.markdown(f"**{emoji} {label}**")
                st.write(response.answer)

                if response.sources:
                    tab1, tab2, tab3 = st.tabs([
                        "📝 Sources",
                        f"📄 Documents ({len(response.doc_chunks)})",
                        f"🌐 Web ({len(response.web_results)})"
                    ])

                    with tab1:
                        for s in response.sources:
                            if s.source_type == "document":
                                st.markdown(f"📄 `{s.citation}`")
                            else:
                                st.markdown(f"🌐 `{s.citation}` — [link]({s.url})")

                    with tab2:
                        if response.doc_chunks:
                            for chunk in response.doc_chunks:
                                with st.expander(f"📄 {chunk['citation']}"):
                                    st.write(chunk['content'])
                                    st.caption(f"Score: {chunk['score']:.3f}")
                        else:
                            st.info("No document chunks used")

                    with tab3:
                        if response.web_results:
                            for r in response.web_results:
                                with st.expander(f"🌐 {r.title}"):
                                    st.write(r.content)
                                    st.markdown(f"[🔗 Full article]({r.url})")
                        else:
                            st.info("No web results used")

                st.session_state.chat_history.append((question, response))

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info(
                    "**Quick fixes:**\n"
                    "- Add your Groq API key in the sidebar\n"
                    "- Add Tavily key if web search is on\n"
                    "- Upload documents before asking doc questions"
                )

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray'>Built with Groq + LangChain + FAISS + Streamlit</div>", unsafe_allow_html=True)