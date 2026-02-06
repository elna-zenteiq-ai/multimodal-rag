"""Streamlit UI for Multimodal RAG System - ChatGPT style."""

import streamlit as st
import requests
import uuid
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f0f0;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #22863a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-citation {
        background-color: #f0f8ff;
        padding: 0.8rem;
        border-left: 3px solid #0066cc;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .file-badge {
        display: inline-block;
        background-color: #e1f5ff;
        color: #01579b;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-ready {
        color: #22863a;
        font-weight: bold;
    }
    .status-processing {
        color: #ff6f00;
        font-weight: bold;
    }
    .status-failed {
        color: #cb2431;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar
with st.sidebar:
    st.title("🤖 Multimodal RAG")
    
    # Current conversation
    st.subheader("Conversation")
    st.code(st.session_state.conversation_id, language="text")
    
    if st.button("🔄 New Conversation"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.rerun()
    
    st.divider()
    
    # File Upload
    st.subheader("📤 Upload Documents")
    st.caption("Max 10 files, 10MB each")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md"],
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("⬆️ Upload", type="primary", use_container_width=True):
            try:
                with st.spinner("Uploading..."):
                    files_data = [
                        ("files", (f.name, f.read(), f.type))
                        for f in uploaded_files
                    ]
                    
                    headers = {
                        "conversation-id": st.session_state.conversation_id
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/upload",
                        files=files_data,
                        headers=headers,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.uploaded_files = result.get("file_ids", [])
                        st.success(f"✅ Uploaded {len(result.get('file_ids', []))} file(s)")
                        st.rerun()
                    else:
                        st.error(f"❌ {response.json().get('detail', 'Upload failed')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.subheader("📂 Uploaded Files")
        
        # Check status
        if st.button("🔍 Check Status"):
            try:
                headers = {"conversation-id": st.session_state.conversation_id}
                response = requests.get(f"{API_BASE_URL}/status", headers=headers)
                
                if response.status_code == 200:
                    status_data = response.json()
                    for file in status_data.get("files", []):
                        status = file["status"]
                        emoji = {"READY": "✅", "PROCESSING": "⏳", "FAILED": "❌"}.get(status, "❓")
                        st.markdown(f"{emoji} **{file['filename']}**")
                        if status != "READY":
                            st.caption(f"Status: {status}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Health Check
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏥 Health", help="Check API health"):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    health = response.json()
                    if health["status"] == "healthy":
                        st.success("✅ Healthy")
                    else:
                        st.warning("⚠️ Degraded")
                else:
                    st.error("❌ Error")
            except Exception:
                st.error("❌ Offline")
    
    with col2:
        if st.button("🗑️ Clear", help="Clear chat"):
            st.session_state.messages = []
            st.rerun()

# Main chat area
st.title("Chat")

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown(f"""
            <div class="user-message">
            <strong>You:</strong><br>
            {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for source in message["sources"]:
                        pages_text = f"Pages: {', '.join(map(str, source.get('page_numbers', [])))}" if source.get('page_numbers') else ""
                        heading_text = f"**{source.get('heading', '')}**" if source.get('heading') else ""
                        
                        st.markdown(f"""
                        <div class="source-citation">
                        <strong>{source.get('filename', 'Document')}</strong><br>
                        {pages_text}<br>
                        {heading_text}<br>
                        <br>
                        <em>{source.get('text', '')[:300]}...</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            if message.get("suggestion"):
                st.info(f"💡 {message['suggestion']}")

# Input section - fixed at bottom
st.divider()

# Chat input
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Ask about your documents or upload new ones...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and user_input:
    try:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show loading state
        with st.spinner("⏳ Thinking..."):
            headers = {"conversation-id": st.session_state.conversation_id}
            
            response = requests.post(
                f"{API_BASE_URL}/send",
                data={"message": user_input},
                headers=headers,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant response
                answer = result.get("answer") or "No response received"
                sources = result.get("sources", [])
                suggestion = result.get("suggestion")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "suggestion": suggestion
                })
                
                st.rerun()
            else:
                error_msg = response.json().get("detail", "An error occurred")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {error_msg}",
                    "sources": [],
                    "suggestion": None
                })
                st.rerun()
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Connection error: {str(e)}",
            "sources": [],
            "suggestion": None
        })
        st.rerun()
elif send_button and not user_input:
    st.warning("Please enter a message")
