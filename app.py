"""
Tokyo Electricity FAQ Assistant - a small NLP question-answering app (Streamlit).

Ask anything about electricity demand in the Tokyo area, the weather and this project's data.
The assistant finds the best answer with TF-IDF + cosine similarity (qa_engine.py), looks up
specific dates in the real data, and politely refuses empty, invalid or off-topic input.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from qa_engine import FAQAssistant

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Tokyo Electricity FAQ Assistant", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

STARTERS = [("When is electricity demand highest during the day?", "Time patterns"),
            ("What was the record peak demand?", "Records"),
            ("Does heat or cold add more demand?", "Weather"),
            ("How hot was it on 5 August 2024?", "Date lookup")]
ORDER = ["Records", "Weather", "Time patterns", "Solar & supply", "Forecasting", "Basics", "About the data", "Tips",
         "About this app", "Trends"]

# Visual style: grid-blue header, red-orange accent, card-like chat bubbles (theme colours: .streamlit/config.toml).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@500;700&display=swap');
.block-container {padding-top: 3.6rem; max-width: 980px;}
[data-testid="stProgress"] div[role="progressbar"] > div > div {background-color: #1f6fb2;}
.app-bar {background: #0e3a5c; color: #f2f7fb; border-radius: 16px; padding: 18px 22px; display: flex;
          align-items: center; gap: 14px; margin-bottom: 18px;}
.app-bar h1 {font-family: 'Zen Kaku Gothic New', sans-serif; font-size: 28px; margin: 0; padding: 0; color: #f2f7fb;
             line-height: 1.15;}
.app-bar p {margin: 4px 0 0; color: #a9c4da; font-size: 15px;}
.app-bar .pill {margin-left: auto; font-size: 13px; color: #a9c4da; border: 1px solid #2b5a80; border-radius: 999px;
                padding: 3px 10px; white-space: nowrap;}
.app-bar .pill b {color: #3ecf8e;}
.logo {flex: none;}
[data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.18); border-radius: 14px; padding: 14px 16px;
                               margin-bottom: 10px;}
.meta {font-size: 13px; opacity: .75; margin-top: 2px;}
.tag {display: inline-block; background: rgba(31,111,178,.12); border-radius: 6px; padding: 1px 8px; margin-right: 6px;}
.hero {text-align: center; padding: 18px 0 6px;}
.hero h2 {font-family: 'Zen Kaku Gothic New', sans-serif; font-size: 30px; margin-bottom: 4px;}
.hero p {opacity: .75; max-width: 560px; margin: 0 auto 14px;}
.acc {font-size: 14px; margin: 2px 0 -6px; display: flex; justify-content: space-between;}
@media (max-width: 640px) {.app-bar {flex-wrap: wrap;} .app-bar .pill {margin-left: 0;} .app-bar h1 {font-size: 22px;}}
</style>
""", unsafe_allow_html=True)

LOGO = ('<svg class="logo" width="44" height="44" viewBox="0 0 34 34"><circle cx="17" cy="17" r="16" fill="#e8452c"/>'
        '<path d="M19 5 9 19h7l-2 10 11-15h-7z" fill="#fff"/></svg>')


@st.cache_resource
def load_assistant() -> FAQAssistant:
    return FAQAssistant()


@st.cache_data
def load_scores():
    p = HERE / "evaluation_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


bot = load_assistant()
st.session_state.setdefault("history", [])


def queue(question: str):
    st.session_state.pending = question


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Browse by topic")
    groups = {}
    for f in bot.faq:
        groups.setdefault(f["category"], []).append(f["question"])
    cats = [c for c in ORDER if c in groups] + [c for c in groups if c not in ORDER]
    for ci, cat in enumerate(cats):
        with st.expander(f"{cat}  ·  {len(groups[cat])}", expanded=(ci == 0)):
            for qi, q in enumerate(groups[cat]):
                st.button(q, key=f"topic{ci}_{qi}", on_click=queue, args=(q,), width="stretch")
    scores = load_scores()
    if scores:
        t = scores["final_test_set"]
        st.markdown("### Model accuracy")
        for label, val in [("Answered right (first try)", t["top1_accuracy_pct"]),
                           ("Right answer in top 3", t["top3_accuracy_pct"]),
                           ("Off-topic questions refused", t["off_topic_rejection_pct"])]:
            st.markdown(f'<div class="acc"><span>{label}</span><b>{val:.1f} %</b></div>', unsafe_allow_html=True)
            st.progress(val / 100)
        st.caption("Held-out test of 42 new questions. TF-IDF (words + character fragments) and cosine similarity "
                   "over 180 phrasings of 33 answers, with an off-topic detector.")
    if st.button("🗑️ New chat", width="stretch"):
        st.session_state.history = []
        st.rerun()
    st.caption("Data © TEPCO Power Grid and the Japan Meteorological Agency (processed), via kaseee/tokyo-grid-lab. "
               "Covers 1 Apr 2022 – 12 Oct 2026.")

# ------------------------------------------------------------------ header
st.markdown(f"""
<div class="app-bar">{LOGO}
  <div><h1>Tokyo Electricity FAQ Assistant</h1>
  <p>Ask about electricity demand in the Tokyo (TEPCO) area, weather, solar power, holidays, or any date.</p></div>
  <span class="pill"><b>●</b> Model ready · {len(bot.faq)} topics</span>
</div>""", unsafe_allow_html=True)

typed = st.chat_input("Ask about Tokyo electricity, or a date like 2024-08-05")
question = typed if typed is not None else st.session_state.pop("pending", None)
if question is not None:
    st.session_state.history.append({"q": question, "r": bot.ask(question)})

# ------------------------------------------------------------------ empty state
if not st.session_state.history:
    st.markdown('<div class="hero"><h2>Ask me about Tokyo\'s electricity</h2><p>Demand records, how heat and cold '
                'change it, solar power, holidays, the forecasting model, or any day between 1 Apr 2022 and '
                '12 Oct 2026.</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (q, cat) in enumerate(STARTERS):
        cols[i % 2].button(q, key=f"start{i}", on_click=queue, args=(q,), width="stretch", help=cat)

# ------------------------------------------------------------------ conversation
for n, turn in enumerate(st.session_state.history):
    r = turn["r"]
    with st.chat_message("user", avatar="🧑"):
        st.markdown(turn["q"] if str(turn["q"]).strip() else "*(empty message)*")
    with st.chat_message("assistant", avatar="⚡"):
        if r.status == "invalid":
            st.warning(r.message, icon="⚠️")
        elif r.status == "unsure":
            st.info(r.message, icon="🤔")
        else:
            st.markdown(r.message)
            tag = "From the daily data" if r.status == "date" else r.category
            matched = "exact date match" if r.status == "date" else f"matched: <i>{r.matched_question}</i>"
            st.markdown(f'<div class="meta"><span class="tag">{tag}</span>{matched} · confidence '
                        f'{r.confidence:.0%}</div>', unsafe_allow_html=True)
            st.progress(min(max(r.confidence, 0.0), 1.0))
        if r.related:
            if r.status == "answer":
                st.markdown('<div class="meta">Related questions</div>', unsafe_allow_html=True)
            cols = st.columns(len(r.related))
            for k, (col, rq) in enumerate(zip(cols, r.related)):
                col.button(rq, key=f"rel{n}_{k}", on_click=queue, args=(rq,), width="stretch")
