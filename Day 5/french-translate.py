import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import google.generativeai as genai

# --- SET YOUR GOOGLE GEMINI API KEY HERE ---
GEMINI_API_KEY = "AIzaSyDEH75dihWolMGMaEBxjbBlOJVhDmH03nE"  # 🔐 Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Gemini Translator", layout="centered")
st.title("🌍 English to French Translator using Gemini + LangChain")

# --- Input Section ---
user_input = st.text_area("✏️ Enter an English sentence to translate:", height=150)
translate_button = st.button("Translate to French")

# --- Action on Button Click ---
if translate_button:
    if not user_input.strip():
        st.error("❌ Please enter a sentence to translate.")
    else:
        try:
            # 1. Define the Gemini LLM
            llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)

            # 2. Define the Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that translates English to French."),
                ("user", "Translate this sentence: {english_input}")
            ])

            # 3. Create a chain using Runnable
            chain: Runnable = prompt | llm

            # 4. Run the chain with input
            result = chain.invoke({"english_input": user_input})

            # 5. Display the result
            st.success("✅ Translation completed!")
            st.markdown("### 🇫🇷 Translated French Sentence:")
            st.info(result.content.strip())

        except Exception as e:
            st.error(f"🚫 Error during translation: {str(e)}")