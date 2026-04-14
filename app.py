import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Memory 🤖")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ✅ FIX 1: Add {chat_history} to prompt so it matches invoke() keys
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based only on the provided context.
Provide the most accurate response.
<context>
{context}
</context>"""),
    MessagesPlaceholder(variable_name="chat_history"),  # ✅ chat history slot
    ("human", "{input}"),
])

if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "📂 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ✅ FIX 2: Use tempfile instead of saving with file.name (breaks on Streamlit Cloud)
def process_pdfs(uploaded_files):
    documents = []
    tmp_paths = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp_paths[-1])
        docs = loader.load()
        documents.extend(docs)

    # ✅ FIX 3: Clean up temp files after loading
    for path in tmp_paths:
        os.unlink(path)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_docs, embeddings)
    return vectors

if st.button("⚡ Process PDFs"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.vectors = process_pdfs(uploaded_files)
        st.success("✅ Vector Store Ready!")
    else:
        st.warning("Please upload PDFs first!")

user_input = st.chat_input("Ask something from your documents...")

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
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    # ✅ FIX 4: Only pass keys that exist in the prompt template
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history  # ✅ now valid
    })

    answer = response["answer"]

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))

    st.chat_message("assistant").write(answer)
    st.caption(f"⏱ Response time: {time.process_time() - start:.2f} sec")

if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()  # ✅ st.success after clear won't show anyway; rerun is cleaner
