"""
Streamlit frontend for the AI chatbot with conversation history sidebar
"""
import streamlit as st
import requests
import json
import re
from typing import List, Dict
import datetime

# API Configuration - update to match the backend
API_URL = "http://127.0.0.1:8080"  # Updated to match backend configuration

# App title and configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Initialize session state variables
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # Dictionary to store multiple conversations
    
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
    
if "next_conv_id" not in st.session_state:
    st.session_state.next_conv_id = 1  # For generating local conversation IDs

# Function to create a new conversation
def create_new_conversation():
    local_id = f"conv_{st.session_state.next_conv_id}"
    st.session_state.next_conv_id += 1
    timestamp = datetime.datetime.now().strftime("%H:%M %p")
    
    st.session_state.conversations[local_id] = {
        "session_id": None,  # Will be set when first message is sent
        "messages": [],
        "title": f"New Chat",
        "timestamp": timestamp
    }
    return local_id

# Create initial conversation if none exists
if not st.session_state.conversations:
    new_id = create_new_conversation()
    st.session_state.current_conversation_id = new_id

# Function to switch conversation
def switch_conversation(conv_id):
    st.session_state.current_conversation_id = conv_id
    st.rerun()

# Function to get conversation title
def get_conversation_title(messages):
    if not messages:
        return "New Chat"
    # Use first user message as title (truncated)
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"]
            if len(title) > 25:
                title = title[:25] + "..."
            return title
    return "New Chat"

# Function to update conversation title
def update_conversation_title():
    conv_id = st.session_state.current_conversation_id
    if conv_id and st.session_state.conversations[conv_id]["messages"]:
        st.session_state.conversations[conv_id]["title"] = get_conversation_title(
            st.session_state.conversations[conv_id]["messages"]
        )

# Updated function to thoroughly clean responses
def clean_response(text):
    """Thoroughly clean any response from the model"""
    if not text:
        return "No response generated. Please try again."
    
    # Remove any instruction tags
    text = re.sub(r'\[/?INST\]', '', text)
    
    # Remove any system instruction blocks
    text = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[SYS\].*?\[/SYS\]', '', text, flags=re.DOTALL)
    
    # Remove any common assistant prefixes 
    prefixes = ["Assistant:", "AI:", "Llama:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove any other tags
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove model tokens
    text = text.replace("</s>", "").strip()
    
    # Remove triple backticks
    text = re.sub(r'^```.*?\n|```$', '', text, flags=re.DOTALL)
    
    # If we end up with nothing after cleaning, provide a fallback
    if not text.strip():
        return "I apologize, but I couldn't generate a proper response. Please try asking your question again."
        
    return text.strip()

# Left sidebar with conversation history
with st.sidebar:
    # New chat button
    if st.button("+ New Chat", use_container_width=True):
        new_id = create_new_conversation()
        st.session_state.current_conversation_id = new_id
        st.rerun()
    
    st.divider()
    
    # Display conversation history
    st.subheader("Chat History")
    
    for conv_id, conv_data in sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True
    ):
        # Create a button for each conversation
        title = conv_data.get("title", "New Chat")
        timestamp = conv_data.get("timestamp", "")
        
        # Highlight current conversation
        if conv_id == st.session_state.current_conversation_id:
            button_style = f"font-weight: bold; background-color: #f0f2f6;"
        else:
            button_style = ""
            
        # Use columns for conversation button and delete button
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"{title}\n{timestamp}", key=f"conv_{conv_id}", use_container_width=True):
                switch_conversation(conv_id)
        
        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                # Delete the conversation from API if it has a session_id
                if st.session_state.conversations[conv_id].get("session_id"):
                    try:
                        session_id = st.session_state.conversations[conv_id]["session_id"]
                        requests.delete(f"{API_URL}/sessions/{session_id}", timeout=5)
                    except:
                        pass
                
                # Delete from local state
                del st.session_state.conversations[conv_id]
                
                # If we deleted the current conversation, switch to another or create new
                if conv_id == st.session_state.current_conversation_id:
                    if st.session_state.conversations:
                        st.session_state.current_conversation_id = next(iter(st.session_state.conversations))
                    else:
                        new_id = create_new_conversation()
                        st.session_state.current_conversation_id = new_id
                
                st.rerun()
    
    # Status indicator and app info (at the bottom of sidebar)
    st.divider()
    st.subheader("API Status")
    if st.button("Check Connection"):
        try:
            with st.spinner("Checking connection..."):
                response = requests.get(f"{API_URL}/", timeout=5)
                if response.status_code == 200:
                    st.success(f"✅ Connected to API")
                else:
                    st.error(f"❌ API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API")
            st.info("Make sure the backend is running")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # App info
    st.divider()
    st.markdown("**AI Chatbot**")
    st.markdown("Powered by:")
    st.markdown("- Amazon Bedrock")
    st.markdown("- LangChain & FastAPI")

# Main chat area
if st.session_state.current_conversation_id:
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    
    # Display header
    st.title("🤖 AI Chatbot")
    
    # Create container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display initial message when no messages exist
        if not current_conv["messages"]:
            st.info("👋 Welcome! Type a message to start chatting with the  AI assistant.")
        
        # Display existing messages
        for message in current_conv["messages"]:
            with st.chat_message(message["role"]):
                # Clean any stored messages just in case
                cleaned_content = clean_response(message["content"])
                st.markdown(cleaned_content)
    
    # Accept user input at the bottom
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        current_conv["messages"].append({"role": "user", "content": prompt})
        
        # Update the conversation title after first message
        if len(current_conv["messages"]) == 1:
            update_conversation_title()
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Prepare request data
                request_data = {
                    "message": prompt
                }
                
                # Include session_id if we have one
                if current_conv.get("session_id"):
                    request_data["session_id"] = current_conv["session_id"]
                    
                # Send request to API with a timeout
                response = requests.post(
                    f"{API_URL}/chat",
                    json=request_data,
                    timeout=30  # Increased timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Save the session ID
                    current_conv["session_id"] = response_data["session_id"]
                    
                    # Clean the response
                    assistant_response = clean_response(response_data["response"])
                    
                    # Update placeholder with response
                    message_placeholder.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    current_conv["messages"].append({"role": "assistant", "content": assistant_response})
                    
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
                    current_conv["messages"].append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to the backend API. Please make sure the backend server is running."
                message_placeholder.error(error_msg)
                current_conv["messages"].append({"role": "assistant", "content": error_msg})
                
            except requests.exceptions.Timeout:
                error_msg = "Request timed out. The backend server might be overloaded or not responding."
                message_placeholder.error(error_msg)
                current_conv["messages"].append({"role": "assistant", "content": error_msg})
                
            except Exception as e:
                error_msg = f"Error communicating with the API: {str(e)}"
                message_placeholder.error(error_msg)
                current_conv["messages"].append({"role": "assistant", "content": error_msg})
