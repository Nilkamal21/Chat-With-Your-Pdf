import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


import os
from dotenv import load_dotenv

load_dotenv()  # loads your .env into os.environ

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GROQ_API_KEY not found in environment or .env file!")

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192")  # will automatically pick up GROQ_API_KEY

# Initialize Hugging Face embeddings (open-source)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("📄 Chat with Your PDF ")

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

    # Step 3: Create or update vector store
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
            st.warning("Please enter a question!")
        else:
            with st.spinner("Thinking..."):
                result = qa({"query": question})
                st.write("**Answer:**", result["result"])

               