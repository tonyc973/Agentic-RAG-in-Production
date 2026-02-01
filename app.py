import streamlit as st
import os
import uuid
import tempfile
import requests
import warnings
import time # <-- Used for the safe-delete buffer
from dotenv import load_dotenv

# --- Production Imports ---
from streamlit_cookies_controller import CookieController
from streamlit_mic_recorder import mic_recorder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Database & Cloud Imports ---
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from pinecone import Pinecone # <-- Native client for deleting vector namespaces

# --- Local Imports ---
from src.crew_graph import AgenticWorkflow

# --- Init ---
load_dotenv()
st.set_page_config(page_title="Agentic Voice RAG", layout="wide", page_icon="🧠")
warnings.filterwarnings("ignore")

# Initialize audio duplicate check
if "last_audio" not in st.session_state: 
    st.session_state.last_audio = None

# ---------------------------------------------------------
# 1. DATABASE UTILITIES & HELPERS
# ---------------------------------------------------------
DB_CONNECTION = os.getenv("DB_CONNECTION")
db_engine = create_engine(DB_CONNECTION)

# Ensure the titles table exists
with db_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            session_id TEXT PRIMARY KEY,
            title TEXT
        )
    """))
    conn.commit()

def get_chat_titles():
    """Fetches all sessions and their AI-generated titles."""
    try:
        with db_engine.connect() as conn:
            query = text("""
                SELECT DISTINCT m.session_id, COALESCE(t.title, 'New Conversation')
                FROM message_store m
                LEFT JOIN chat_titles t ON m.session_id = t.session_id
                ORDER BY m.session_id DESC
            """)
            result = conn.execute(query)
            return {row[0]: row[1] for row in result}
    except ProgrammingError:
        return {}

def generate_title(s_id, first_query):
    """Uses LLM to summarize the first user query into a 4-word title."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        title = llm.invoke(f"Summarize this query into a 4-word title: {first_query}").content.replace('"', '')
        
        with db_engine.connect() as conn:
            # 🔴 CRITICAL FIX: Add "ON CONFLICT" clause
            query = text("""
                INSERT INTO chat_titles (session_id, title) 
                VALUES (:s, :t)
                ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title
            """)
            conn.execute(query, {"s": s_id, "t": title})
            conn.commit()
    except Exception as e:
        print(f"Title generation skipped: {e}")

def delete_session(s_id):
    """BULLETPROOF DELETE: Erases PostgreSQL text history AND Pinecone vectors."""
    # 1. Erase PostgreSQL Database History
    with db_engine.connect() as conn:
        conn.execute(text("DELETE FROM message_store WHERE session_id = :s"), {"s": s_id})
        conn.execute(text("DELETE FROM chat_titles WHERE session_id = :s"), {"s": s_id})
        conn.commit()
    
    # 2. Erase Pinecone Vector Namespace
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("agentic-rag-prod")
        index.delete(delete_all=True, namespace=s_id)
    except Exception:
        # Fails silently if the user never uploaded a PDF to this specific chat
        pass

def transcribe_audio(audio_bytes):
    """Sends audio to your on-premise WhisperCPP Docker container."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            fname = f.name
        with open(fname, "rb") as f:
            url = os.getenv("WHISPER_URL", "http://audio_backend:5001/transcribe")
            resp = requests.post(url, files={"audio": (os.path.basename(fname), f, "audio/wav")}, timeout=30)
        os.unlink(fname)
        return resp.json().get("transcription") if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Audio Server Error: {e}")
        return None

# ---------------------------------------------------------
# 2. SESSION MANAGEMENT
# ---------------------------------------------------------
controller = CookieController()
st.sidebar.title("🧠 Agentic RAG")
session_id = controller.get('user_session_id')

if not session_id:
    session_id = str(uuid.uuid4())
    controller.set('user_session_id', session_id)

# ---------------------------------------------------------
# 3. INITIALIZE STATE (PostgreSQL + Pinecone)
# ---------------------------------------------------------
chat_history = SQLChatMessageHistory(
    session_id=session_id,
    connection_string=DB_CONNECTION
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name="agentic-rag-prod", 
    embedding=embeddings, 
    namespace=session_id # CRITICAL: Isolates user data
)

# ---------------------------------------------------------
# 4. SIDEBAR UI (Chat History & Uploads)
# ---------------------------------------------------------
chat_titles_map = get_chat_titles()

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        controller.set('user_session_id', new_id)
        st.rerun()

    st.divider()
    st.header("🕒 Chat History")
    
    # Render the chat list with auto-titles and safe-delete buttons
    for s_id, title in chat_titles_map.items():
        col_chat, col_del = st.columns([5, 1])
        label = f"💬 {title}"
        if s_id == session_id: label = f"👉 {title}"
            
        with col_chat:
            if st.button(label, key=f"chat_{s_id}", use_container_width=True):
                controller.set('user_session_id', s_id)
                st.rerun()
                
        with col_del:
            if st.button("🗑️", key=f"del_{s_id}", help="Delete Chat & PDFs"):
                delete_session(s_id)
                # CRITICAL FIX: If active chat is deleted, sleep allows cookie to save before refresh
                if s_id == session_id:
                    controller.set('user_session_id', str(uuid.uuid4()))
                    time.sleep(0.5) 
                st.rerun()

    st.divider()
    st.header("⚙️ Add Knowledge")
    uploaded_file = st.file_uploader("Upload PDF to this Chat", type="pdf")
    
    if uploaded_file and st.button("Ingest Knowledge"):
        with st.spinner("Indexing to Secure Cloud VectorDB..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            vectorstore.add_documents(chunks)
            st.success(f"Indexed {len(chunks)} chunks securely.")
            os.unlink(tmp_path)

# ---------------------------------------------------------
# 5. MAIN CHAT UI (VOICE + TEXT)
# ---------------------------------------------------------
st.title("🧠 Production Agentic Chat")

def get_recent_history():
    messages = chat_history.messages[-6:]
    return "\n".join([f"{m.type}: {m.content}" for m in messages])

# --- ADD THE MIC TO THE SIDEBAR ---
with st.sidebar:
    st.divider()
    st.header("🎙️ Voice Assistant")
    audio = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", key='rec', format='wav')

# --- RENDER CHAT HISTORY ---
# This naturally pushes everything up so the chat input stays at the bottom
for msg in chat_history.messages:
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role): st.write(msg.content)

# --- NATIVE CHAT INPUT (Anchored to the bottom) ---
user_input = st.chat_input("Type message or use the mic in the sidebar...")

final_query = None

# Audio Check (From Sidebar)
if audio and audio['bytes'] != st.session_state.last_audio:
    st.session_state.last_audio = audio['bytes']
    with st.spinner("Transcribing audio locally..."):
        transcribed_text = transcribe_audio(audio['bytes'])
        if transcribed_text: final_query = transcribed_text

# Text Check (From Bottom Input)
if user_input:
    final_query = user_input

# Logic Execution
if final_query:
    chat_history.add_user_message(final_query)
    with st.chat_message("user"): st.write(final_query)
    
    # Smart Titling
    is_first_message = len(chat_history.messages) <= 2
    if is_first_message and session_id not in chat_titles_map:
        generate_title(session_id, final_query)
    
    # Run Agentic Workflow
    with st.chat_message("assistant"):
        with st.spinner("Agents are researching..."):
            try:
                workflow = AgenticWorkflow(vector_db=vectorstore)
                response = workflow.run(query=final_query, chat_history=get_recent_history())
                st.write(response)
                chat_history.add_ai_message(response)
            except Exception as e:
                st.error(f"Error: {e}")
    is_first_message = len(chat_history.messages) <= 2
    if is_first_message and session_id not in chat_titles_map:
        generate_title(session_id, final_query)
        st.rerun()