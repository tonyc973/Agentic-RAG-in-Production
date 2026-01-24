import streamlit as st
import os
import uuid
import tempfile
import requests
import warnings
from dotenv import load_dotenv

# --- Production Imports ---
from streamlit_cookies_controller import CookieController
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Local Imports ---
from src.crew_graph import AgenticWorkflow

# --- Init ---
load_dotenv()
st.set_page_config(page_title="Agentic Voice RAG (Prod)", layout="wide", page_icon="🧠")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. SESSION MANAGEMENT (Cookies)
# ---------------------------------------------------------
controller = CookieController()
session_id = controller.get('user_session_id')

if not session_id:
    session_id = str(uuid.uuid4())
    controller.set('user_session_id', session_id)

st.sidebar.caption(f"Session ID: {session_id[:8]}...")

# ---------------------------------------------------------
# 2. PERSISTENT CHAT HISTORY (PostgreSQL)
# ---------------------------------------------------------
chat_history = SQLChatMessageHistory(
    session_id=session_id,
    connection_string=os.getenv("DB_CONNECTION")
)

# ---------------------------------------------------------
# 3. SECURE VECTOR STORE (Pinecone)
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "agentic-rag-prod"

vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings, 
    namespace=session_id # CRITICAL: Isolates user data
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload Knowledge (PDF)", type="pdf")
    if uploaded_file and st.button("Ingest Knowledge"):
        with st.spinner("Indexing to Cloud VectorDB..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            
            # Upsert vectors into the user's secure namespace
            vectorstore.add_documents(chunks)
            st.success(f"Indexed {len(chunks)} chunks securely.")
            os.unlink(tmp_path)

# --- Helper ---
def get_recent_history():
    messages = chat_history.messages[-6:]
    return "\n".join([f"{m.type}: {m.content}" for m in messages])

# --- Main UI ---
st.title("🧠 Production Agentic Template")

# Render chat from DATABASE
for msg in chat_history.messages:
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role): st.write(msg.content)

# Inputs
user_input = st.chat_input("Type message...")

if user_input:
    chat_history.add_user_message(user_input)
    with st.chat_message("user"): st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Agent Crew is working..."):
            try:
                # Pass namespaced vectorstore and DB history to your workflow
                workflow = AgenticWorkflow(vector_db=vectorstore)
                response = workflow.run(query=user_input, chat_history=get_recent_history())
                
                st.write(response)
                chat_history.add_ai_message(response)
            except Exception as e:
                st.error(f"Error: {e}")