import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load environment variables
load_dotenv()

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit Title
st.title("Gemma Model Document Q&A Chatbot")

# LLM setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
Provide the most accurate response.

Context:
{context}

Question:
{input}
""")

# Function to create vector store
def create_vector_store():
    if "vectors" not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        loader = PyPDFDirectoryLoader("./Docs")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
        st.success("Vector Store DB is Ready")

# UI to trigger vector store creation
if st.button("Create Vector Store"):
    create_vector_store()

# User input
prompt1 = st.text_input("Ask a question based on the documents:")

# Response generation
if prompt1 and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})

    st.write("### Answer:")
    st.write(response['answer'])

    with st.expander("View Relevant Documents"):
        for doc in response['context']:
            st.write(doc.page_content)
            st.write("––––––––––––––––––––––––––")
