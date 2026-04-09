import streamlit as st
import os
import time

from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
##from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import HumanMessage, AIMessage

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Memory 🤖")

# Initialize LLM (Groq)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    Provide the most accurate response.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Session state init
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_files = st.file_uploader(
    "📂 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# Function: Create vector store
def process_pdfs(uploaded_files):
    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        documents.extend(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_docs = text_splitter.split_documents(documents)

    vectors = FAISS.from_documents(final_docs, embeddings)
    return vectors

# Button to process PDFs
if st.button("⚡ Process PDFs"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.vectors = process_pdfs(uploaded_files)
        st.success("✅ Vector Store Ready!")
    else:
        st.warning("Please upload PDFs first!")

# Chat input
user_input = st.chat_input("Ask something from your documents...")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)


if user_input:
    if st.session_state.vectors is None:
        st.error("⚠️ Please upload and process PDFs first!")
        st.stop()

    st.chat_message("user").write(user_input)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    answer = response["answer"]

    
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))

    st.chat_message("assistant").write(answer)

    st.caption(f"⏱ Response time: {time.process_time() - start:.2f} sec")


if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared!")