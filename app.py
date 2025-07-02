import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1:1.5b",  # your Ollama model
    
)

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

               
