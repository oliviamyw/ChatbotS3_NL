# =========================
# Style Loom Chatbot Experiment
# STUDY 3 - VISUAL PRESENT × LOW AUTONOMY
# =========================

import os
import re
import sys
import uuid
import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# chromadb on Streamlit Cloud
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from openai import OpenAI
from supabase import create_client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Style Loom Chatbot Experiment", layout="centered")


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# -------------------------
# Experiment constants
# -------------------------
MODEL_CHAT = "gpt-4o-mini"
MODEL_EMBED = "text-embedding-3-small"
MIN_USER_TURNS = 5
TBL_SESSIONS = "sessions"


# -------------------------
# OpenAI client
# -------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

client = OpenAI(api_key=API_KEY)


# -------------------------
# Supabase client
# -------------------------
SUPA_URL = st.secrets.get("SUPABASE_URL", None)
SUPA_KEY = st.secrets.get("SUPABASE_ANON_KEY", None)

if not SUPA_URL or not SUPA_KEY:
    st.error("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(SUPA_URL, SUPA_KEY)


supabase = get_supabase()


# -------------------------
# Study 3 cell condition
# -------------------------
identity_option = "No name and no image"
show_name = False
show_picture = False
CHATBOT_NAME = "Skyler"
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png"

autonomy_condition = "Low autonomy"


def chatbot_speaker() -> str:
    return CHATBOT_NAME if show_name else "Style Loom Assistant"


# -------------------------
# Header UI
# -------------------------
st.markdown(
    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;'>"
    "<div style='font-weight:700;font-size:20px;letter-spacing:0.3px;'>Style Loom</div>"
    "</div>",
    unsafe_allow_html=True,
)

if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=84)
    except Exception:
        pass


# -------------------------
# Issues
# -------------------------
ISSUES = [
    "— Select an issue —",
    "Wrong item received",
    "Delivery delay",
    "Damaged or defective item",
    "Return or refund issue",
    "Payment issue",
    "Order cancellation",
    "Other",
]

ISSUE_TO_HINT = {
    "Wrong item received": "wrong item received order mismatch refund replacement",
    "Delivery delay": "delivery delay shipping issue delayed shipment expected delivery tracking",
    "Damaged or defective item": "damaged defective item refund replacement product issue",
    "Return or refund issue": "return refund process friction return request refund status",
    "Payment issue": "payment issue payment declined checkout error order not confirmed payment status",
    "Order cancellation": "order cancellation cancel order before shipment order status",
    "Other": "other issue customer service support shopping help",
    "— Select an issue —": "",
}


# -------------------------
# Load vectorstore from ./data/*.txt
# -------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(data_dir: Path) -> Optional[Chroma]:
    try:
        if not data_dir.exists():
            return None

        loader = DirectoryLoader(
            str(data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=False,
        )

        docs = loader.load()

        if not docs:
            return None

        for d in docs:
            src = d.metadata.get("source", "")
            d.metadata["filename"] = os.path.basename(src)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
        )
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            model=MODEL_EMBED,
            openai_api_key=API_KEY,
        )

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="styleloom_study3_visualpresent_highautonomy",
        )

        return vectordb

    except Exception as e:
        st.warning(f"Vectorstore could not be built: {e}")
        return None


vectorstore = build_vectorstore(DATA_DIR)


def retrieve_context(query: str, k: int = 8, min_score: float = 0.20) -> str:
    if not vectorstore:
        return ""

    try:
        hits = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        filtered = [(d, s) for (d, s) in hits if s is not None and s >= min_score]

        if not filtered:
            return ""

        blocks = []
        for i, (d, s) in enumerate(filtered, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} score={s:.2f} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)

    except Exception:
        hits = vectorstore.similarity_search(query, k=k)
        if not hits:
            return ""

        blocks = []
        for i, d in enumerate(hits, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)


def load_all_txt_as_fallback() -> str:
    if not DATA_DIR.exists():
        return ""

    blocks = []
    for fp in sorted(DATA_DIR.glob("*.txt")):
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            content = ""

        if content:
            blocks.append(f"[FILE: {fp.name}]\n{content}")

    return "\n\n".join(blocks)


# -------------------------
# LLM helpers
# -------------------------
def llm_chat(messages: List[dict], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def format_recent_history(chat_history: List[Tuple[str, str]], limit: int = 6) -> str:
    if not chat_history:
        return ""

    turns = chat_history[-limit:]
    lines = []
    for spk, msg in turns:
        role = "User" if spk == "User" else chatbot_speaker()
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


def extract_last_question(text_block: str) -> Optional[str]:
    if not text_block:
        return None

    txt = re.sub(r"\s+", " ", text_block).strip()
    if "?" not in txt:
        return None

    parts = re.split(r"(?<=[\?])\s+", txt)
    qs = [p.strip() for p in parts if p.strip().endswith("?")]
    return qs[-1] if qs else None


def answer_fallback() -> str:
    return "Please briefly describe what happened so the issue can be reviewed and the next steps can be provided."


# -------------------------
# High-autonomy response generation
# -------------------------
def answer_grounded(
    user_text: str,
    context: str,
    issue_label: Optional[str] = None,
    recent_history: str = "",
    pending_question: Optional[str] = None,
) -> str:

    system = f"""
You are Style Loom's virtual assistant in a controlled customer-service study.

This condition is LOW AUTONOMY.
Describe actions as if decisions are made by the system or company policies, not by you.

Use low-autonomy phrasing such as:
- Our system reviewed
- The system reviewed
- The system processed
- Company policy indicates
- According to company procedures
- Based on store policy
- The request can be submitted
- The issue can be reviewed
- Support can be provided
- Assistance can be provided

Do not use first-person decision-making expressions such as:
- I decided
- I reviewed
- I approved
- I processed
- I can help
- I can guide you

Variation rule:
- Maintain a low-autonomy style throughout the conversation.
- Do not repeat the exact same system-centered phrase in consecutive responses when a natural alternative is available.
- Vary low-autonomy wording across turns while keeping the same meaning and tone.
- Keep the language professional, clear, and consistent.
- Do not make the wording overly varied or stylistically dramatic.

Use the BUSINESS CONTEXT as the source of truth for store procedures and issue-handling guidance.

Rules:
- Keep the response concise, natural, and professional.
- Do not use echoing expressions like "Understood" or "Got it."
- Do not repeat the user's wording unnecessarily.
- Do not claim that the issue has already been resolved.
- Provide procedural guidance only.
- Ask at most one follow-up question if necessary.
- Keep tone, sentence length, and fluency stable.
- Current issue category: {issue_label or "unknown"}.
"""

    msgs = [{"role": "system", "content": system}]

    if recent_history.strip():
        msgs.append({"role": "system", "content": f"RECENT CHAT:\n{recent_history}"})

    if pending_question:
        msgs.append({"role": "system", "content": f"PREVIOUS ASSISTANT QUESTION:\n{pending_question}"})

    if context.strip():
        msgs.append({"role": "system", "content": f"BUSINESS CONTEXT:\n{context}"})

    msgs.append({"role": "user", "content": user_text})

    return llm_chat(msgs, temperature=0.2)


def generate_answer(user_text: str, issue: Optional[str]) -> str:
    issue_hint = ISSUE_TO_HINT.get(issue or "", "")
    query = f"{issue or ''} {issue_hint} {user_text}".strip()

    recent_history = format_recent_history(st.session_state.get("chat_history", []), limit=6)
    pending_q = st.session_state.get("pending_question")

    context = ""
    if vectorstore:
        context = retrieve_context(query, k=8, min_score=0.20)

    if not context.strip():
        context = load_all_txt_as_fallback()

    if not context.strip():
        return answer_fallback()

    return answer_grounded(
        user_text=user_text,
        context=context,
        issue_label=issue,
        recent_history=recent_history,
        pending_question=pending_q,
    )


# -------------------------
# Session state
# -------------------------
defaults = {
    "chat_history": [],
    "session_id": str(uuid.uuid4()),
    "greeted_once": False,
    "ended": False,
    "rating_saved": False,
    "user_turns": 0,
    "bot_turns": 0,
    "last_user_selected_issue": "— Select an issue —",
    "active_issue": None,
    "pending_question": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------------
# Greeting
# -------------------------
if (not st.session_state.greeted_once) and (not st.session_state.chat_history):
    greet_text = "Hi, I’m Style Loom’s virtual assistant. I’m here to help with your shopping."
    st.session_state.chat_history.append((chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# -------------------------
# Issue selectbox
# -------------------------
st.markdown("**How can I help you today?**")

selected = st.selectbox(
    "Choose an issue:",
    options=ISSUES,
    index=ISSUES.index(st.session_state.last_user_selected_issue)
    if st.session_state.last_user_selected_issue in ISSUES else 0,
)

prev_selected = st.session_state.last_user_selected_issue
st.session_state.last_user_selected_issue = selected

if selected != "— Select an issue —" and selected != prev_selected:
    st.session_state.active_issue = selected
    confirm_text = f"Please describe what happened with **{selected}** so the issue can be reviewed."
    st.session_state.chat_history.append((chatbot_speaker(), confirm_text))

st.divider()


# -------------------------
# Render chat history
# -------------------------
for spk, msg in st.session_state.chat_history:
    if spk == chatbot_speaker():
        st.markdown(f"**{spk}:** {msg}")
    else:
        st.markdown(f"**User:** {msg}")


# -------------------------
# Chat input
# -------------------------
user_text = None
if not st.session_state.ended:
    user_text = st.chat_input("Type your message here...")


# -------------------------
# End button and rating
# -------------------------
end_col1, end_col2 = st.columns([1, 2])

with end_col1:
    can_end = (st.session_state.user_turns >= MIN_USER_TURNS) and (not st.session_state.ended)
    if st.button("End chat", disabled=not can_end):
        st.session_state.ended = True

with end_col2:
    if not st.session_state.ended:
        completed = st.session_state.user_turns
        remaining = max(0, MIN_USER_TURNS - completed)

        if remaining > 0:
            st.caption(
                f"Please complete at least {MIN_USER_TURNS} user turns before ending the chat. "
                f"Progress: {completed}/{MIN_USER_TURNS}."
            )
        else:
            st.caption(f"Progress: {completed}/{MIN_USER_TURNS}. You can end the chat now.")


# -------------------------
# Save only at the end
# -------------------------
if st.session_state.ended and not st.session_state.rating_saved:
    rating = st.slider("Overall satisfaction with the chatbot (1 = very low, 7 = very high)", 1, 7, 4)
    prolific_id = st.text_input("Prolific ID", value="")

    if st.button("Submit rating and save"):
        ts_now = datetime.datetime.utcnow().isoformat()

        final_issue = st.session_state.active_issue or (
            selected if selected != "— Select an issue —" else "Other"
        )

        transcript_lines = []
        transcript_lines.append("===== Session Transcript =====")
        transcript_lines.append(f"timestamp       : {ts_now}")
        transcript_lines.append(f"session_id      : {st.session_state.session_id}")
        transcript_lines.append(f"identity_option : {identity_option}")
        transcript_lines.append(f"autonomy        : {autonomy_condition}")
        transcript_lines.append(f"picture_present : {'present' if show_picture else 'absent'}")
        transcript_lines.append(f"issue           : {final_issue}")
        transcript_lines.append(f"name_present    : {'present' if show_name else 'absent'}")
        transcript_lines.append(f"user_turns      : {st.session_state.user_turns}")
        transcript_lines.append(f"bot_turns       : {st.session_state.bot_turns}")
        transcript_lines.append(f"prolific_id     : {(prolific_id.strip() or 'N/A')}")
        transcript_lines.append("")
        transcript_lines.append("---- Chat transcript ----")
        for spk, msg in st.session_state.chat_history:
            transcript_lines.append(f"{spk}: {msg}")
        transcript_lines.append("")
        transcript_lines.append(f"Satisfaction (1-7): {int(rating)}")

        transcript_text = "\n".join(transcript_lines)

        session_payload = {
            "session_id": st.session_state.session_id,
            "ts_start": ts_now,
            "ts_end": ts_now,
            "identity_option": identity_option,
            "autonomy_condition": autonomy_condition,
            "name_present": "present" if show_name else "absent",
            "picture_present": "present" if show_picture else "absent",
            "issue": final_issue,
            "user_turns": st.session_state.user_turns,
            "bot_turns": st.session_state.bot_turns,
            "prolific_id": prolific_id.strip() or None,
            "transcript": transcript_text,
            "satisfaction": int(rating),
        }
        
        try:
            supabase.table(TBL_SESSIONS).insert(session_payload).execute()
            st.session_state.rating_saved = True
            st.success("Saved. Thank you.")
        except Exception as e:
            st.write(e)

# -------------------------
# Main interaction
# -------------------------
if user_text and not st.session_state.ended:
    st.session_state.chat_history.append(("User", user_text))
    st.session_state.user_turns += 1

    active_issue = st.session_state.active_issue or (
        selected if selected != "— Select an issue —" else "Other"
    )

    answer = generate_answer(user_text, issue=active_issue)
    st.session_state["pending_question"] = extract_last_question(answer)

    st.session_state.chat_history.append((chatbot_speaker(), answer))
    st.session_state.bot_turns += 1

    st.rerun()

