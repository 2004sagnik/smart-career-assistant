"""
app.py — Streamlit UI for Smart Career Assistant
Run: streamlit run app.py
"""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Smart Career Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg0:#0d1117; --bg1:#161c26; --bg2:#1c2333; --bg3:#222d40;
    --gold:#e6a817; --gold2:#c48a0a; --teal:#38bdf8;
    --green:#4ade80; --red:#f87171; --amber:#fbbf24;
    --txt1:#e2e8f0; --txt2:#94a3b8; --txt3:#4b5563; --border:#1e2d45;
}
html,body,[class*="css"]{font-family:'Sora',sans-serif!important;background:var(--bg0)!important;color:var(--txt1)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
.block-container{padding-top:1rem!important;}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"]>div{padding:1.5rem 1rem;}
.logo{font-size:1.45rem;font-weight:700;color:var(--gold);}
.logo-sub{font-size:0.72rem;color:var(--txt2);margin:2px 0 1.2rem;letter-spacing:.4px;}
.divider{border:none;border-top:1px solid var(--border);margin:.9rem 0;}
.sec-lbl{font-size:.68rem;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;}
.chip{display:inline-block;background:var(--bg2);border:1px solid var(--border);border-left:2px solid var(--teal);color:var(--txt2);font-size:.7rem;padding:3px 8px;border-radius:4px;margin:2px 1px;}
.ubox{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:6px;padding:8px 12px;font-size:.77rem;color:var(--txt2);margin-bottom:.8rem;}
.ubox b{color:var(--gold);}
.stButton>button{background:linear-gradient(135deg,var(--gold),var(--gold2))!important;color:#0d1117!important;border:none!important;border-radius:7px!important;font-family:'Sora',sans-serif!important;font-weight:600!important;font-size:.82rem!important;padding:9px 16px!important;width:100%!important;}
.cwrap{max-width:800px;margin:0 auto;padding:0 1rem 5rem;}
.ptitle{font-size:1.85rem;font-weight:700;color:var(--txt1);text-align:center;padding:1.5rem 0 .3rem;letter-spacing:-.5px;}
.psub{text-align:center;color:var(--txt2);font-size:.84rem;margin-bottom:1.6rem;}
.mrow{display:flex;gap:10px;margin-bottom:1rem;align-items:flex-start;}
.mrow.usr{flex-direction:row-reverse;}
.av{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.88rem;font-weight:700;flex-shrink:0;}
.av-u{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;}
.av-b{background:linear-gradient(135deg,var(--gold),var(--gold2));color:var(--bg0);}
.bub{padding:11px 15px;border-radius:10px;max-width:80%;font-size:.86rem;line-height:1.7;white-space:pre-wrap;word-break:break-word;}
.b-u{background:var(--bg3);border:1px solid #2a3d5c;border-top-right-radius:2px;color:var(--txt1);}
.b-b{background:var(--bg2);border:1px solid var(--border);border-top-left-radius:2px;color:var(--txt1);}
.meta{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 0 44px;}
.tag{font-size:.67rem;padding:2px 7px;border-radius:12px;font-weight:500;font-family:'JetBrains Mono',monospace;}
.t-rt{background:#0d2035;color:var(--teal);border:1px solid #0d3050;}
.t-fh{background:#0d2a1a;color:var(--green);border:1px solid #0d3520;}
.t-fm{background:#2a2000;color:var(--amber);border:1px solid #3a2d00;}
.t-fl{background:#2a0d0d;color:var(--red);border:1px solid #3a1515;}
.t-sc{background:#131826;color:var(--txt2);border:1px solid var(--border);}
.empty{text-align:center;padding:3rem 2rem;color:var(--txt3);font-size:.87rem;line-height:1.8;}
.stTextInput>div>div>input{background:var(--bg2)!important;border:1px solid var(--border)!important;color:var(--txt1)!important;border-radius:8px!important;padding:11px 14px!important;font-family:'Sora',sans-serif!important;font-size:.86rem!important;}
.stTextInput>div>div>input:focus{border-color:var(--gold)!important;box-shadow:0 0 0 2px rgba(230,168,23,.12)!important;}
.fnote{font-size:.67rem;color:var(--txt3);text-align:center;margin-top:.4rem;}
</style>
""", unsafe_allow_html=True)


def init():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_goal" not in st.session_state:
        st.session_state.user_goal = ""

init()


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    from knowledge_base import get_rag_components
    from graph import get_graph
    get_rag_components()
    get_graph()
    return True

load_resources()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo">🎓 Career Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">For B.Tech Students · Placements &amp; Govt Exams</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">About</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:.77rem;color:#64748b;line-height:1.7;'>Agentic AI built with LangGraph, ChromaDB RAG, and Gemini/Groq. Answers are strictly grounded — no hallucination.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Topics Covered</div>', unsafe_allow_html=True)
    topics = ["SSC CHSL Pattern","SSC Typing Test","SSC Syllabus","Banking Exams",
              "DSA Roadmap","Aptitude Strategy","Resume Tips","Interview Prep",
              "Govt vs IT Job","Study Planning","Time Management","Internship Prep",
              "Placement Timeline","Common Mistakes"]
    st.markdown("".join(f'<span class="chip">{t}</span>' for t in topics), unsafe_allow_html=True)

    if st.session_state.user_name or st.session_state.user_goal:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        parts = []
        if st.session_state.user_name: parts.append(f"<b>{st.session_state.user_name}</b>")
        if st.session_state.user_goal: parts.append(st.session_state.user_goal)
        st.markdown(f'<div class="ubox">👤 {" &nbsp;·&nbsp; ".join(parts)}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history   = []
        st.session_state.user_name = ""
        st.session_state.user_goal = ""
        st.rerun()
    st.markdown("<p style='font-size:.67rem;color:#374151;text-align:center;margin-top:.8rem;'>LangGraph · ChromaDB · Gemini/Groq</p>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="cwrap">', unsafe_allow_html=True)
st.markdown('<div class="ptitle">Smart Career Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="psub">Ask about SSC, banking exams, placements, DSA, or career planning.</div>', unsafe_allow_html=True)


def faith_tag(s):
    if s >= 0.8:   return f'<span class="tag t-fh">✓ {s:.2f}</span>'
    elif s >= 0.6: return f'<span class="tag t-fm">~ {s:.2f}</span>'
    else:          return f'<span class="tag t-fl">⚠ {s:.2f}</span>'


if not st.session_state.history:
    st.markdown('<div class="empty">Try asking:<br><em>"What is the SSC CHSL exam pattern?"</em><br><em>"Give me a DSA preparation roadmap."</em><br><em>"Should I choose a government job or IT job?"</em></div>', unsafe_allow_html=True)
else:
    for e in st.session_state.history:
        init_char = (e.get("user_name") or "U")[0].upper()
        st.markdown(f'<div class="mrow usr"><div class="av av-u">{init_char}</div><div class="bub b-u">{e["question"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mrow"><div class="av av-b">🎓</div><div class="bub b-b">{e["answer"]}</div></div>', unsafe_allow_html=True)
        tags = f'<span class="tag t-rt">⇢ {e.get("route","")}</span>'
        tags += faith_tag(e.get("faithfulness", 1.0))
        for s in (e.get("sources") or [])[:2]:
            short = (s[:25]+"…") if len(s)>25 else s
            tags += f'<span class="tag t-sc">📄 {short}</span>'
        st.markdown(f'<div class="meta">{tags}</div>', unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        user_input = st.text_input("Question", placeholder="Ask about placements, SSC, banking, DSA...", label_visibility="collapsed")
    with c2:
        submitted = st.form_submit_button("Send →")

if submitted and user_input and user_input.strip():
    with st.spinner("Thinking..."):
        from graph import ask
        result = ask(user_input.strip(), thread_id=st.session_state.thread_id)
    if result.get("user_name"): st.session_state.user_name = result["user_name"]
    if result.get("user_goal"): st.session_state.user_goal = result["user_goal"]
    st.session_state.history.append({
        "question":    user_input.strip(),
        "answer":      result["answer"],
        "route":       result["route"],
        "faithfulness":result["faithfulness"],
        "sources":     result["sources"],
        "user_name":   st.session_state.user_name,
    })
    st.rerun()

st.markdown('<div class="fnote">Answers grounded strictly in the knowledge base. No hallucination by design.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
