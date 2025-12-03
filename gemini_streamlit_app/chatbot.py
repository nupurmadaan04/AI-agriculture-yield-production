# chatbot.py
import streamlit as st
from datetime import datetime
from gemini_service import ask_gemini  # your wrapper for Gemini API

# --- App config ---
st.set_page_config(page_title="AgriGemini Chatbot", layout="centered")

# --- Helpers ---
def format_message(sender: str, msg: str):
    time = datetime.now().strftime("%H:%M")
    return f"**{sender}** • {time}\n\n{msg}"

def agriculture_system_prompt():
    return (
        "You are an expert Agriculture Advisor (short name: AgriBot). "
        "Answer clearly and concisely for farmers and agronomists. "
        "Focus on practical steps and local context. "
        "For yield predictions, request numeric inputs (area, production, year, soil type, irrigation, fertilizer). "
        "For technical/model details, give short summaries and offer code/notebooks. "
        "Always provide safety tips and suggest consulting local experts for critical actions."
    )

# --- Sidebar ---
st.sidebar.title("AgriGemini Chatbot")
st.sidebar.markdown("Crop Monitoring & Yield Prediction assistant")

# Supported Gemini models (free-tier friendly)
model_choice = st.sidebar.selectbox(
    "Model",
    ("gemini-2.5-flash", "gemini-2.5-pro")
)

max_tokens = st.sidebar.slider("Max output tokens (approx)", 128, 1024, 512, step=64)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("🔒 Ensure your API key is set as `GEMINI_API_KEY` in environment variables.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- Header ---
st.title("🌾 AgriGemini — Crop Monitoring & Yield Prediction")
st.markdown("Ask about crop care, yield estimation, soil tips, or model questions.")

# --- Quick Prompts ---
with st.expander("Quick Prompts (click to use)"):
    c1, c2, c3 = st.columns(3)
    if c1.button("How to increase rice yield?"):
        st.session_state.input_text = "How can I increase rice yield in monsoon conditions with paddy fields?"
    if c2.button("Interpreting model results"):
        st.session_state.input_text = "Explain the model's evaluation metrics and what they mean for farm decision-making."
    if c3.button("Fertilizer schedule"):
        st.session_state.input_text = "Give a fertilizer schedule for rice on loamy soil."

# --- Chat Area ---
chat_col, sidebar_col = st.columns([3, 1])
with chat_col:
    # Display chat history
    for entry in st.session_state.messages:
        sender = entry["sender"]
        text = entry["text"]
        color = "#0B5FFF" if sender == "User" else "#0B7E3F"
        align = "right" if sender == "User" else "left"
        st.markdown(f"<div style='text-align:{align};color:{color}'>{format_message(sender, text)}</div>", unsafe_allow_html=True)

    # Input form
    with st.form("input_form", clear_on_submit=False):
        user_input = st.text_input("Your question", value=st.session_state.input_text)
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({"sender": "User", "text": user_input})

            # Build prompt
            system_prompt = agriculture_system_prompt()
            final_prompt = f"{system_prompt}\n\nUser Question:\n{user_input}\n\nAnswer concisely:"

            # Typing indicator
            placeholder = st.empty()
            placeholder.info("AgriBot is typing...")

            # Call Gemini API
            reply = ask_gemini(
                final_prompt,
                model_name=model_choice,
                max_output_tokens=max_tokens
            )

            # Replace typing indicator with bot reply
            placeholder.empty()
            st.session_state.messages.append({"sender": "AgriBot", "text": reply})

            # Clear input_text for next message
            st.session_state.input_text = ""

# --- Sidebar controls ---
with sidebar_col:
    st.markdown("### Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("### Export")
    if st.button("Download chat (txt)"):
        all_text = "\n\n".join([f"{m['sender']}: {m['text']}" for m in st.session_state.messages])
        st.download_button(
            label="Download conversation",
            data=all_text,
            file_name="agri_chat.txt",
            mime="text/plain"
        )

# --- Footer / Tips ---
st.markdown("---")
st.markdown(
    "**Tips:** Provide numeric details (area, year, production, soil type) for better yield predictions. "
    "Always verify critical farm decisions with local agronomists."
)
