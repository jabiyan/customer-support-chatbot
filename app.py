import streamlit as st
from rag_pipeline import build_rag_pipeline, get_answer
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Support Chatbot",
    page_icon="🛒",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.chat-user {
    background-color: #7F77DD;
    color: white;
    padding: 10px 15px;
    border-radius: 15px 15px 0px 15px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
}
.chat-bot {
    background-color: #f0f2f6;
    color: #1a1a1a;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0px;
    margin: 5px 0;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    placeholder="Paste your API key here"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Sample Questions")
sample_questions = [
    "How long does shipping take?",
    "What is your return policy?",
    "How do I track my order?",
    "What payment methods do you accept?",
    "I received a damaged item, what do I do?",
    "How do I reset my password?",
    "Do you have a loyalty program?",
]
for q in sample_questions:
    if st.sidebar.button(q, use_container_width=True):
        st.session_state.prefill = q

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Mustajab Hussain**")
st.sidebar.markdown(
    "[LinkedIn](https://www.linkedin.com/in/mustajab-hussain-312475283/) | "
    "[GitHub](https://github.com/jabiyan)"
)

# ── Header ────────────────────────────────────────────────────
st.title("🛒 E-Commerce Support Chatbot")
st.markdown(
    "Ask me anything about **shipping, returns, payments, orders, or your account**. "
    "I'm here to help 24/7."
)
st.markdown("---")

# ── Initialize session state ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── Build RAG pipeline ────────────────────────────────────────
if api_key and st.session_state.rag_chain is None:
    with st.spinner("🔧 Setting up AI assistant... (first time only, ~30 seconds)"):
        try:
            st.session_state.rag_chain = build_rag_pipeline(api_key)
            st.success("✅ Assistant ready! Ask me anything.")
        except Exception as e:
            st.error(f"Error setting up pipeline: {e}")

elif not api_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to start.")
    st.stop()

# ── Chat history display ──────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">👤 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bot">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ── Chat input ────────────────────────────────────────────────
prefill_val = st.session_state.prefill
st.session_state.prefill = ""

user_input = st.chat_input(
    "Ask a question about orders, shipping, returns...",
)

# Handle sidebar button prefill
if prefill_val and not user_input:
    user_input = prefill_val

# ── Process question ──────────────────────────────────────────
if user_input and st.session_state.rag_chain:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get RAG answer
    with st.spinner("🤔 Thinking..."):
        try:
            result   = get_answer(st.session_state.rag_chain, user_input)
            answer   = result["answer"]
            sources  = result["sources"]

            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}. Please try again."
            })

    st.rerun()

# ── Clear chat button ─────────────────────────────────────────
if st.session_state.messages:
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
