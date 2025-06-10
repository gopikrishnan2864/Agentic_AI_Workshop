import streamlit as st
import google.generativeai as genai
from fpdf import FPDF

# --- Set your Gemini API Key here ---
GEMINI_API_KEY = "AIzaSyBu_YKRNvE_hyRNN1Z7BtDYzuXnKp54vRs"
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini model setup ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- PDF Export Function ---
def generate_pdf(email_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in email_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf

# --- Streamlit UI ---
st.title("📧 AI-Powered Email Generator using Gemini")

# Input Section
user_input = st.text_area("✏️ Enter your main points or message", height=200)

# Email Format and Tone
format_option = st.selectbox("🧾 Select Email Format", ["Formal", "Informal", "Business", "Apology", "Follow-up"])
tone_option = st.radio("🎭 Choose Tone", ["Professional", "Friendly", "Urgent", "Polite", "Confident"])

# Session state for storing output
if 'email_output' not in st.session_state:
    st.session_state.email_output = ""

# Generate/Regenerate Button
if st.button("🚀 Generate Email"):
    if user_input.strip() == "":
        st.warning("Please enter some input text.")
    else:
        prompt = (
            f"Write a {tone_option.lower()} and {format_option.lower()} email based on the following message:\n"
            f"{user_input}"
        )
        response = model.generate_content(prompt)
        st.session_state.email_output = response.text
        st.success("Email generated successfully!")

# Display Email
if st.session_state.email_output:
    st.subheader("📨 Generated Email")
    st.text_area("Email Output", value=st.session_state.email_output, height=300)

    # Download as PDF
    if st.download_button(
        label="📥 Download as PDF",
        data=generate_pdf(st.session_state.email_output).output(dest='S').encode('latin-1'),
        file_name="generated_email.pdf",
        mime="application/pdf",
    ):
        st.success("Downloaded PDF!")

    # Regenerate Option
    if st.button("🔁 Regenerate with New Format/Tone"):
        st.session_state.email_output = ""
