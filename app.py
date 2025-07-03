import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import shutil
import streamlit as st
from dotenv import load_dotenv

# 🧹 Clear vector store directory BEFORE anything touches it
if os.path.exists("./db"):
    try:
        shutil.rmtree("./db")
    except Exception as e:
        print(f"❌ Failed to delete ./db: {e}")

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file!")

# Initialize your Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",  # Replace with another supported model if needed
    groq_api_key=groq_api_key,
)

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("📄 Chat with Your PDF")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to disk temporarily
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully. Building knowledge base...")

    # Step 2: Load & split PDF
    loader = PyPDFLoader("uploaded.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Step 3: Create new vector store
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./db")
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Step 4: Set up RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Step 5: User asks questions
    st.info("Your PDF is ready! Start asking questions below 👇")
    question = st.text_input("🔎 Your question:")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("⚠️ Please enter a question!")
        else:
            with st.spinner("🤔 Thinking..."):
                result = qa({"query": question})
                st.write("**Answer:**", result["result"])

                # Optionally show sources (uncomment if needed)
                # for doc in result["source_documents"]:
                #     st.write(f"📄 **Source page:**\n{doc.page_content[:500]}...")
