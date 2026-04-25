import streamlit as st
import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import tempfile
import time

st.set_page_config(page_title="StudyMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ══ FULL APP DARK ══ */
.stApp { background: #0d1117 !important; }
.main .block-container { padding: 1.6rem 2rem 3rem !important; max-width: 1140px; background: #0d1117 !important; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d !important; }
[data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem !important; }
[data-testid="stSidebar"] * { color: #8b949e !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e6edf3 !important; font-weight: 700 !important; }
[data-testid="stSidebar"] hr { border-color: #21262d !important; margin: 0.8rem 0 !important; }

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] { background: #161b22 !important; border: 2px dashed #30363d !important; border-radius: 12px !important; }
[data-testid="stFileUploader"] * { color: #8b949e !important; }
[data-testid="stFileUploader"] button { background: #21262d !important; color: #e6edf3 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }

/* ══ BUTTONS — bright, solid, impossible to miss ══ */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    transition: all 0.18s !important;
    box-shadow: 0 0 0 1px rgba(240,246,252,0.1), 0 4px 12px rgba(35,134,54,0.35) !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover { background: #2ea043 !important; transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(46,160,67,0.45) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ══ DOWNLOAD BUTTON — blue ══ */
[data-testid="stDownloadButton"] > button { background: #1f6feb !important; color: #fff !important; box-shadow: 0 4px 12px rgba(31,111,235,0.35) !important; }
[data-testid="stDownloadButton"] > button:hover { background: #388bfd !important; box-shadow: 0 6px 20px rgba(56,139,253,0.45) !important; }

/* ══ CHAT INPUT ══ */
[data-testid="stChatInput"] textarea { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; color: #e6edf3 !important; font-size: 0.95rem !important; }
[data-testid="stChatInput"] textarea:focus { border-color: #388bfd !important; box-shadow: 0 0 0 3px rgba(56,139,253,0.15) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #484f58 !important; }

/* ══ CHAT MESSAGES ══ */
[data-testid="stChatMessage"] { border-radius: 14px !important; padding: 0.9rem 1.1rem !important; margin: 0.5rem 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #1c2128 !important; border: 1px solid #388bfd !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: #161b22 !important; border: 1px solid #21262d !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span { color: #e6edf3 !important; line-height: 1.75 !important; }

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] { background: #161b22 !important; border-radius: 12px !important; padding: 4px !important; gap: 3px !important; border: 1px solid #21262d !important; }
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #8b949e !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.5rem 1.2rem !important; transition: all 0.15s !important; }
.stTabs [aria-selected="true"] { background: #238636 !important; color: #ffffff !important; box-shadow: 0 2px 10px rgba(35,134,54,0.4) !important; }

/* ══ METRICS ══ */
[data-testid="metric-container"] { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 14px !important; padding: 1rem 1.2rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.8rem !important; font-weight: 800 !important; }

/* ══ PROGRESS ══ */
.stProgress > div > div { background: linear-gradient(90deg, #238636, #3fb950) !important; border-radius: 99px !important; }
.stProgress > div { background: #21262d !important; border-radius: 99px !important; }

/* ══ SLIDER ══ */
[data-testid="stSlider"] * { color: #8b949e !important; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: #c9d1d9 !important; }

/* ══ EXPANDER ══ */
.streamlit-expanderHeader { background: #161b22 !important; border-radius: 10px !important; color: #e6edf3 !important; font-weight: 600 !important; border: 1px solid #21262d !important; }
.streamlit-expanderContent { background: #0d1117 !important; border: 1px solid #21262d !important; border-top: none !important; }
.streamlit-expanderContent p, .streamlit-expanderContent li { color: #c9d1d9 !important; }
.streamlit-expanderContent strong { color: #e6edf3 !important; }

/* ══ ALERTS ══ */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ══ ALL MAIN TEXT ══ */
.main p, .main li { color: #c9d1d9 !important; line-height: 1.75 !important; }
.main h1 { color: #e6edf3 !important; font-weight: 800 !important; }
.main h2, .main h3 { color: #c9d1d9 !important; font-weight: 700 !important; }
.main strong { color: #e6edf3 !important; }
.main code { background: #161b22 !important; color: #79c0ff !important; border: 1px solid #21262d !important; border-radius: 5px !important; padding: 2px 6px !important; }

/* ════════════ CUSTOM COMPONENTS ════════════ */

/* Hero */
.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 60%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 18px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero::after {
    content: ''; position: absolute; top: -80px; right: -40px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(35,134,54,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero-icon  { font-size: 2.8rem; flex-shrink: 0; }
.hero-title { font-size: 2rem; font-weight: 800; color: #e6edf3; margin: 0 0 5px; letter-spacing: -0.5px; }
.hero-sub   { font-size: 0.875rem; color: #8b949e; margin: 0; line-height: 1.6; }
.hero-pill  {
    margin-left: auto; flex-shrink: 0;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; padding: 0.4rem 1.1rem;
    font-size: 0.76rem; color: #8b949e; font-weight: 600; white-space: nowrap;
}
.hero-pill.loaded { background: rgba(35,134,54,0.2); border-color: #238636; color: #3fb950; }

/* Status pills */
.pill-ok   { display:inline-block; background:rgba(35,134,54,0.2); border:1px solid #238636; color:#3fb950; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:700; }
.pill-wait { display:inline-block; background:rgba(187,128,9,0.2); border:1px solid #bb8009; color:#d29922; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:700; }

/* File items */
.fitem { display:flex; align-items:center; gap:8px; background:#161b22; border:1px solid #21262d; border-radius:10px; padding:7px 11px; margin:4px 0; }
.fdot  { width:7px; height:7px; border-radius:50%; background:#3fb950; flex-shrink:0; }
.fname { font-size:0.79rem; color:#c9d1d9; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fext  { font-size:0.66rem; background:#21262d; color:#8b949e; padding:2px 7px; border-radius:6px; font-weight:700; border:1px solid #30363d; }

/* Sidebar label */
.slabel { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#484f58 !important; margin:0.8rem 0 0.4rem; display:block; }

/* Brand */
.brand     { display:flex; align-items:center; gap:10px; padding:0.2rem 0 0.8rem; }
.brand-ico { font-size:1.8rem; }
.brand-nm  { font-size:1.05rem; font-weight:800; color:#e6edf3 !important; }
.brand-ver { font-size:0.63rem; color:#484f58 !important; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; }

/* Summary box */
.sum-box { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:0.9rem 1rem; font-size:0.82rem; color:#8b949e; line-height:1.65; margin-top:0.5rem; }

/* Empty state */
.empty     { text-align:center; padding:4rem 2rem; }
.empty-ico { font-size:3.5rem; display:block; margin-bottom:1rem; }
.empty-t   { font-size:1.05rem; color:#c9d1d9 !important; font-weight:700; margin-bottom:0.5rem; }
.empty-s   { font-size:0.85rem; color:#484f58 !important; }

/* Quiz card */
.qcard { background:#161b22; border:1px solid #21262d; border-radius:16px; padding:1.6rem 1.9rem; margin-bottom:1.1rem; box-shadow:0 4px 16px rgba(0,0,0,0.3); }
.qnum  { font-size:0.7rem; font-weight:700; color:#3fb950; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.6rem; }
.qtext { font-size:1.05rem; font-weight:700; color:#e6edf3; line-height:1.65; }

/* Quiz options */
.qopt { display:flex; align-items:center; gap:12px; background:#0d1117; border:1px solid #21262d; border-radius:11px; padding:0.78rem 1rem; margin:0.42rem 0; font-size:0.92rem; color:#c9d1d9; }
.qltr { width:28px; height:28px; border-radius:50%; background:#21262d; border:1px solid #30363d; display:flex; align-items:center; justify-content:center; font-size:0.74rem; font-weight:800; color:#8b949e; flex-shrink:0; }
.qcorrect { border-color:#238636 !important; background:rgba(35,134,54,0.15) !important; color:#3fb950 !important; }
.qcorrect .qltr { background:rgba(35,134,54,0.3) !important; border-color:#238636 !important; color:#3fb950 !important; }
.qwrong   { border-color:#da3633 !important; background:rgba(218,54,51,0.12) !important; color:#ff7b72 !important; }
.qwrong .qltr   { background:rgba(218,54,51,0.3) !important; border-color:#da3633 !important; color:#ff7b72 !important; }

/* Result card */
.rcard { border-radius:18px; padding:2.2rem; text-align:center; margin:1rem 0; }
.r-ex  { background:rgba(35,134,54,0.15); border:1px solid #238636; }
.r-ok  { background:rgba(187,128,9,0.15); border:1px solid #bb8009; }
.r-no  { background:rgba(218,54,51,0.12); border:1px solid #da3633; }
.rpct  { font-size:3.6rem; font-weight:800; color:#e6edf3; line-height:1; }
.rmsg  { font-size:1rem; color:#8b949e; margin-top:0.5rem; font-weight:500; }

/* Flashcard front */
.fc-f {
    background: linear-gradient(140deg, #161b22 0%, #1c2128 40%, #0d419d 100%);
    border: 1px solid #1f6feb;
    border-radius: 20px; padding: 3.2rem 2.5rem;
    text-align:center; min-height:220px;
    display:flex; flex-direction:column; justify-content:center; align-items:center;
    box-shadow:0 20px 60px rgba(31,111,235,0.2);
    margin:0.5rem 0 1.1rem; position:relative; overflow:hidden;
}
.fc-f::before { content:''; position:absolute; top:-50%; left:-40%; width:200%; height:200%; background:radial-gradient(circle at 30% 30%, rgba(56,139,253,0.1) 0%, transparent 55%); pointer-events:none; }

/* Flashcard back */
.fc-b {
    background: linear-gradient(140deg, #0d1117 0%, #1c2128 40%, #033a16 100%);
    border: 1px solid #238636;
    border-radius: 20px; padding: 3.2rem 2.5rem;
    text-align:center; min-height:220px;
    display:flex; flex-direction:column; justify-content:center; align-items:center;
    box-shadow:0 20px 60px rgba(35,134,54,0.2);
    margin:0.5rem 0 1.1rem; position:relative; overflow:hidden;
}
.fc-b::before { content:''; position:absolute; top:-55%; right:-30%; width:200%; height:200%; background:radial-gradient(circle at 70% 28%, rgba(63,185,80,0.08) 0%, transparent 55%); pointer-events:none; }

.fc-badge { font-size:0.62rem; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.4); margin-bottom:0.9rem; padding:0.28rem 0.85rem; border:1px solid rgba(255,255,255,0.12); border-radius:20px; }
.fc-text  { font-size:1.5rem; font-weight:800; color:#e6edf3; line-height:1.4; }
.fc-hint  { font-size:0.7rem; color:rgba(255,255,255,0.3); margin-top:0.9rem; }

/* Source chips */
.srow  { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.schip { display:inline-flex; align-items:center; gap:4px; background:#161b22; border:1px solid #21262d; color:#8b949e; font-size:0.72rem; padding:3px 10px; border-radius:20px; font-weight:600; }

/* Notes */
.notes-box { background:#161b22; border:1px solid #21262d; border-radius:16px; padding:1.8rem 2.2rem; box-shadow:0 4px 16px rgba(0,0,0,0.3); }
.notes-box h2 { color:#e6edf3 !important; font-size:1.1rem !important; border-bottom:1px solid #21262d; padding-bottom:0.4rem; margin:1.2rem 0 0.5rem; }
.notes-box h3 { color:#c9d1d9 !important; font-size:1rem !important; margin:1rem 0 0.35rem; }
.notes-box p, .notes-box li { color:#8b949e !important; line-height:1.8; }
.notes-box ul { padding-left:1.3rem; }
.notes-box strong { color:#c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────
defaults = {
    "vector_store": None, "messages": [], "api_key": GROQ_API_KEY,
    "notes": "", "files_processed": False, "quiz_data": [],
    "quiz_index": 0, "quiz_answered": [], "quiz_score": 0,
    "flashcards": [], "flash_index": 0, "flash_flipped": False,
    "doc_summary": "", "processed_filenames": [],
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── HELPERS ────────────────────────────────────────────
def get_llm():
    return ChatGroq(groq_api_key=st.session_state.api_key, model_name="llama-3.3-70b-versatile")

def get_retriever(k=4):
    if st.session_state.vector_store:
        return st.session_state.vector_store.as_retriever(search_kwargs={"k": k})
    return None

def format_sources(docs):
    chips, seen = [], set()
    for d in docs:
        page  = d.metadata.get("page", None)
        src   = d.metadata.get("source", "")
        fname = os.path.basename(src) if src else "document"
        label = f"pg {page+1} · {fname}" if page is not None else fname
        if label not in seen: seen.add(label); chips.append(label)
    return chips

def _ctx(r, q): return "\n\n".join(d.page_content for d in r.invoke(q))

# ── FILE PROCESSING ────────────────────────────────────
def process_files(files):
    if not files: st.warning("⚠️ Pehle files upload karo!"); return
    prog = st.progress(0, text="Starting...")
    emb  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    spl  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_all, total = [], len(files)
    for i, f in enumerate(files):
        prog.progress(i/total, text=f"Processing: {f.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        try:
            loader = PyPDFLoader(path) if f.name.lower().endswith(".pdf") else UnstructuredPowerPointLoader(path)
            docs_all.extend(spl.split_documents(loader.load())); os.remove(path)
        except Exception as e:
            st.error(f"❌ {f.name}: {e}")
            if os.path.exists(path): os.remove(path)
    prog.progress(0.92, text="Building index...")
    time.sleep(0.2)
    if docs_all:
        st.session_state.vector_store        = Chroma.from_documents(docs_all, embedding=emb)
        st.session_state.files_processed     = True
        st.session_state.processed_filenames = [f.name for f in files]
        for k in ["messages","quiz_data","flashcards","quiz_answered"]: st.session_state[k] = []
        for k in ["quiz_index","quiz_score","flash_index"]:              st.session_state[k] = 0
        for k in ["notes","doc_summary"]:                                st.session_state[k] = ""
        st.session_state.flash_flipped = False
        prog.empty(); st.success(f"✅ {total} file(s) ready!")
    else:
        prog.empty(); st.error("❌ No valid documents found.")

# ── LLM FUNCTIONS ──────────────────────────────────────
def generate_notes():
    r = get_retriever(6)
    if not r: return "Pehle files process karo."
    p = PromptTemplate.from_template("Create detailed study notes with ## headings and - bullet points.\n\nContext:\n{context}")
    return (p | get_llm()).invoke({"context": _ctx(r, "summarize all important topics")}).content

def generate_quiz(n=5):
    r = get_retriever(6)
    if not r: return []
    p = PromptTemplate.from_template(
        'Generate exactly {num_q} MCQs. Return ONLY valid JSON array, no markdown.\n'
        '[{{"question":"...","options":["A","B","C","D"],"answer":0}}]\n\nContext:\n{context}')
    raw = (p | get_llm()).invoke({"context": _ctx(r, "important facts, concepts"), "num_q": n}).content
    raw = raw.replace("```json","").replace("```","").strip()
    try: return json.loads(raw)
    except:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    st.error("Quiz error. Try again."); return []

def generate_flashcards(n=10):
    r = get_retriever(6)
    if not r: return []
    p = PromptTemplate.from_template(
        'Create exactly {num} flashcards. Return ONLY JSON array, no markdown.\n'
        '[{{"term":"...","definition":"..."}}]\n\nContext:\n{context}')
    raw = (p | get_llm()).invoke({"context": _ctx(r, "key terms, definitions"), "num": n}).content
    raw = raw.replace("```json","").replace("```","").strip()
    try: return json.loads(raw)
    except:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    st.error("Flashcard error. Try again."); return []

def generate_summary():
    r = get_retriever(5)
    if not r: return ""
    p = PromptTemplate.from_template("Write a 3-4 sentence summary of main topics.\n\nContext:\n{context}")
    return (p | get_llm()).invoke({"context": _ctx(r, "overall summary")}).content

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <span class="brand-ico">🧠</span>
        <div><div class="brand-nm">StudyMind AI</div><div class="brand-ver">v2.0 · Groq Powered</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.files_processed:
        n = len(st.session_state.processed_filenames)
        st.markdown(f'<span class="pill-ok">✓ {n} File{"s" if n>1 else ""} Ready</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for f in st.session_state.processed_filenames:
            ext = f.rsplit(".",1)[-1].upper()
            st.markdown(f'<div class="fitem"><div class="fdot"></div><span class="fname">{f}</span><span class="fext">{ext}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-wait">⏳ No Files Loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="slabel">Upload Files</span>', unsafe_allow_html=True)
    files = st.file_uploader("PDF ya PPTX", type=["pdf","pptx"], accept_multiple_files=True, label_visibility="collapsed")
    if files: st.caption(f"{len(files)} file(s) selected")
    if st.button("⚡  Process Files"): process_files(files)

    st.markdown("---")
    st.markdown('<span class="slabel">Document Overview</span>', unsafe_allow_html=True)
    if st.button("📋  Summary Generate Karo"):
        if not st.session_state.files_processed: st.warning("Pehle files process karo!")
        else:
            with st.spinner("Reading..."): st.session_state.doc_summary = generate_summary()
    if st.session_state.doc_summary:
        st.markdown(f'<div class="sum-box">{st.session_state.doc_summary}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="slabel">Study Notes</span>', unsafe_allow_html=True)
    if st.button("✨  Notes Generate Karo"):
        if not st.session_state.files_processed: st.warning("Pehle files process karo!")
        else:
            with st.spinner("Writing..."): st.session_state.notes = generate_notes()
    if st.session_state.notes:
        st.download_button("⬇️  Download Notes", data=st.session_state.notes, file_name="study_notes.txt", mime="text/plain")

    st.markdown("---")
    if st.button("🗑️  Clear Chat"): st.session_state.messages = []; st.rerun()
    st.markdown('<div style="text-align:center;font-size:0.68rem;color:#21262d;margin-top:0.5rem;">Made with ♥ · Groq + LangChain</div>', unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────
n_files = len(st.session_state.processed_filenames)
pill_cls = "loaded" if st.session_state.files_processed else ""
badge    = f"✓ {n_files} File{'s' if n_files!=1 else ''} Loaded" if st.session_state.files_processed else "Upload Files to Begin"

st.markdown(f"""
<div class="hero">
    <div class="hero-icon">🧠</div>
    <div>
        <div class="hero-title">StudyMind AI</div>
        <div class="hero-sub">PDF &amp; PPTX se chat karo &nbsp;·&nbsp; Quiz lo &nbsp;·&nbsp; Flashcards banao &nbsp;·&nbsp; Notes generate karo</div>
    </div>
    <div class="hero-pill {pill_cls}">{badge}</div>
</div>""", unsafe_allow_html=True)

if st.session_state.files_processed:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📄 Files",    n_files)
    c2.metric("💬 Messages", len(st.session_state.messages))
    ans = len(st.session_state.quiz_answered)
    c3.metric("🎯 Quiz",     f"{st.session_state.quiz_score}/{ans}" if ans else "—")
    c4.metric("🃏 Cards",    len(st.session_state.flashcards) if st.session_state.flashcards else "—")
    st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬  Chat", "🎯  Quiz", "🃏  Flashcards", "📝  Notes"])

# ─── CHAT ───────────────────────────────────
with tab1:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty">
            <span class="empty-ico">💬</span>
            <div class="empty-t">Apne documents se baat karo</div>
            <div class="empty-s">Koi bhi question pucho — source page automatically cite hoga</div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown('<div class="srow">' + "".join(f'<span class="schip">📄 {s}</span>' for s in msg["sources"]) + '</div>', unsafe_allow_html=True)

    if query := st.chat_input("Documents ke baare mein kuch bhi pucho..."):
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            ret = get_retriever()
            if not ret:
                st.warning("⚠️ Pehle files upload aur process karo!")
            else:
                with st.spinner("Thinking..."):
                    llm = get_llm()
                    prompt = ChatPromptTemplate.from_messages([
                        ("system","You are a helpful study assistant. Answer ONLY from context. Be clear and concise.\n\nContext: {context}"),
                        ("human","{input}")
                    ])
                    res = create_retrieval_chain(ret, create_stuff_documents_chain(llm, prompt)).invoke({"input":query})
                answer  = res["answer"]
                sources = format_sources(res.get("context",[]))
                st.markdown(answer)
                if sources:
                    st.markdown('<div class="srow">' + "".join(f'<span class="schip">📄 {s}</span>' for s in sources) + '</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})

    if st.session_state.messages:
        txt = "\n\n".join(f"{'You' if m['role']=='user' else 'AI'}: {m['content']}" for m in st.session_state.messages)
        st.download_button("⬇️  Export Chat", data=txt, file_name="chat_history.txt", mime="text/plain")

# ─── QUIZ ────────────────────────────────────
with tab2:
    ca, cb = st.columns([3,1])
    with ca: num_q = st.slider("Kitne questions?", 3, 10, 5)
    with cb:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲  Quiz Banao"):
            if not st.session_state.files_processed: st.warning("Pehle files process karo!")
            else:
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(num_q)
                if quiz:
                    st.session_state.quiz_data=quiz; st.session_state.quiz_index=0
                    st.session_state.quiz_answered=[]; st.session_state.quiz_score=0; st.rerun()

    if st.session_state.quiz_data:
        tq = len(st.session_state.quiz_data); ans = len(st.session_state.quiz_answered); sc = st.session_state.quiz_score
        st.progress(ans/tq, text=f"Progress: {ans}/{tq}  ·  Score: {sc}/{ans if ans else 0}")

        if ans == tq:
            pct = int(sc/tq*100)
            cls = "r-ex" if pct>=80 else ("r-ok" if pct>=50 else "r-no")
            msg = "🎉 Excellent!" if pct>=80 else ("👍 Accha kiya!" if pct>=50 else "📚 Dobara padho!")
            st.markdown(f'<div class="rcard {cls}"><div class="rpct">{pct}%</div><div class="rmsg">{msg} ({sc}/{tq} correct)</div></div>', unsafe_allow_html=True)
            if st.button("🔄  Restart"):
                st.session_state.quiz_index=0; st.session_state.quiz_answered=[]; st.session_state.quiz_score=0; st.rerun()
        else:
            idx = st.session_state.quiz_index
            q   = st.session_state.quiz_data[idx]
            already = next((a for a in st.session_state.quiz_answered if a["idx"]==idx), None)
            st.markdown(f'<div class="qcard"><div class="qnum">Question {idx+1} of {tq}</div><div class="qtext">{q["question"]}</div></div>', unsafe_allow_html=True)
            for i, opt in enumerate(q["options"]):
                L = chr(65+i)
                if already:
                    if i==q["answer"]:   st.markdown(f'<div class="qopt qcorrect"><div class="qltr">{L}</div> ✅ {opt}</div>', unsafe_allow_html=True)
                    elif i==already["chosen"]: st.markdown(f'<div class="qopt qwrong"><div class="qltr">{L}</div> ❌ {opt}</div>', unsafe_allow_html=True)
                    else:                st.markdown(f'<div class="qopt"><div class="qltr">{L}</div> {opt}</div>', unsafe_allow_html=True)
                else:
                    if st.button(f"{L})  {opt}", key=f"opt_{idx}_{i}"):
                        correct = (i==q["answer"])
                        st.session_state.quiz_answered.append({"idx":idx,"chosen":i,"correct":correct})
                        if correct: st.session_state.quiz_score+=1
                        st.rerun()
            if already:
                p1,p2 = st.columns(2)
                with p1:
                    if idx>0:
                        if st.button("⬅️  Pichla"): st.session_state.quiz_index-=1; st.rerun()
                with p2:
                    if idx<tq-1:
                        if st.button("Agla  ➡️"): st.session_state.quiz_index+=1; st.rerun()
    else:
        st.markdown('<div class="empty"><span class="empty-ico">🎯</span><div class="empty-t">Quiz ready nahi hai</div><div class="empty-s">Files process karo phir "Quiz Banao" dabao</div></div>', unsafe_allow_html=True)

# ─── FLASHCARDS ──────────────────────────────
with tab3:
    cx, cy = st.columns([3,1])
    with cx: num_fc = st.slider("Kitne flashcards?", 5, 20, 10)
    with cy:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨  Cards Banao"):
            if not st.session_state.files_processed: st.warning("Pehle files process karo!")
            else:
                with st.spinner("Making flashcards..."):
                    cards = generate_flashcards(num_fc)
                if cards:
                    st.session_state.flashcards=cards; st.session_state.flash_index=0; st.session_state.flash_flipped=False; st.rerun()

    if st.session_state.flashcards:
        cards=st.session_state.flashcards; idx=st.session_state.flash_index
        flipped=st.session_state.flash_flipped; card=cards[idx]
        st.progress((idx+1)/len(cards), text=f"Card {idx+1} / {len(cards)}")
        if not flipped:
            st.markdown(f'<div class="fc-f"><div class="fc-badge">📖 Term</div><div class="fc-text">{card["term"]}</div><div class="fc-hint">Flip Card dabao definition dekhne ke liye</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fc-b"><div class="fc-badge">💡 Definition</div><div class="fc-text" style="font-size:1.12rem;">{card["definition"]}</div><div class="fc-hint">Flip Card dabao wapas jaane ke liye</div></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            if idx>0:
                if st.button("⬅️  Pichla"): st.session_state.flash_index-=1; st.session_state.flash_flipped=False; st.rerun()
        with c2:
            if st.button("🔄  Flip Card"): st.session_state.flash_flipped=not flipped; st.rerun()
        with c3:
            if idx<len(cards)-1:
                if st.button("Agla  ➡️"): st.session_state.flash_index+=1; st.session_state.flash_flipped=False; st.rerun()
        with st.expander("📋 Saare cards dekho"):
            for i,c in enumerate(cards): st.markdown(f"**{i+1}. {c['term']}** — {c['definition']}")
        st.download_button("⬇️  Download Flashcards", data="\n".join(f"Q: {c['term']}\nA: {c['definition']}\n" for c in cards), file_name="flashcards.txt", mime="text/plain")
    else:
        st.markdown('<div class="empty"><span class="empty-ico">🃏</span><div class="empty-t">Koi flashcard nahi hai</div><div class="empty-s">Files process karo phir "Cards Banao" dabao</div></div>', unsafe_allow_html=True)

# ─── NOTES ───────────────────────────────────
with tab4:
    if st.session_state.notes:
        st.markdown(f'<div class="notes-box">{st.session_state.notes}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        n1,n2 = st.columns(2)
        with n1: st.download_button("⬇️  Download Notes", data=st.session_state.notes, file_name="study_notes.txt", mime="text/plain")
        with n2:
            if st.button("🔄  Regenerate"):
                with st.spinner("Updating..."): st.session_state.notes=generate_notes()
                st.rerun()
    else:
        st.markdown('<div class="empty"><span class="empty-ico">📝</span><div class="empty-t">Notes generate nahi hue</div><div class="empty-s">Sidebar mein "Notes Generate Karo" button dabao</div></div>', unsafe_allow_html=True)`