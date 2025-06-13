import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import tempfile
import torch

# Set page configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Streamlit UI
st.title("📄 PDF Chatbot")
st.write("Upload a PDF file and ask questions about its content.")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process uploaded PDF
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split text into chunks, ensure non-empty chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        texts = [doc for doc in text_splitter.split_documents(documents) if doc.page_content.strip()]
        
        if not texts:
            st.error("No valid text extracted from the PDF. Please ensure the PDF contains text.")
            os.unlink(tmp_file_path)
            st.stop()

        # Create embeddings using HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Initialize local LLM (distilgpt2 for lightweight testing)
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Reduced for stability
            truncation=True,
            device=0 if torch.cuda.is_available() else -1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Initialize conversation chain with limited retrieval
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2}),  # Limit to 2 documents
            memory=memory
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.success("PDF processed successfully! You can now ask questions.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Chat interface
st.subheader("Chat with your PDF")
user_query = st.text_input("Ask a question about the PDF:", key="user_query")

if user_query and st.session_state.qa_chain:
    try:
        # Get response from the QA chain
        result = st.session_state.qa_chain({"question": user_query})
        response = result["answer"] if result["answer"] else "Sorry, I couldn't find relevant information in the PDF."
        
        # Update chat history
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", response))
    except Exception as e:
        st.error(f"Error generating response: {str(e)}. Try rephrasing your question or uploading a different PDF.")

# Display chat history
if st.session_state.chat_history:
    st.write("**Conversation History**")
    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {text}")
        else:
            st.markdown(f"**{speaker}:** {text}")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()