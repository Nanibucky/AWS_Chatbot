"""
Streamlit frontend for the AI chatbot
"""
import streamlit as st
import requests
import json
import time
from typing import List

# API Configuration - update to match the backend
API_URL = "http://127.0.0.1:8080"  # Updated to match backend configuration

# App title and configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar with app information
with st.sidebar:
    st.title("Llama 3 Chatbot")
    st.markdown("This chatbot is powered by:")
    st.markdown("- Amazon Bedrock (Llama 3)")
    st.markdown("- LangChain")
    st.markdown("- FastAPI")
    st.markdown("- Streamlit")
    
    # Add connectivity check
    st.subheader("API Status")
    if st.button("Check API Connection"):
        try:
            with st.spinner("Checking connection..."):
                # Set a short timeout for the request
                response = requests.get(f"{API_URL}/", timeout=5)
                if response.status_code == 200:
                    st.success(f"✅ Connected to API at {API_URL}")
                else:
                    st.error(f"❌ API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API at {API_URL}")
            st.info("Make sure the backend is running and the URL is correct")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Add a clear button
    if st.button("Clear Conversation"):
        if "session_id" in st.session_state:
            session_id = st.session_state.session_id
            try:
                requests.delete(f"{API_URL}/sessions/{session_id}", timeout=5)
                st.session_state.pop("session_id", None)
                st.session_state.messages = []
                st.success("Conversation cleared!")
            except Exception as e:
                st.error(f"Error clearing conversation: {str(e)}")
        else:
            st.session_state.messages = []
            st.success("Conversation cleared!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
st.title("🤖 Llama 3 AI Chatbot")

# Display initial message when no messages exist
if not st.session_state.messages:
    st.info("👋 Welcome! Type a message to start chatting with the Llama 3 AI assistant.")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare request data
                request_data = {
                    "message": prompt
                }
                
                # Include session_id if we have one
                if "session_id" in st.session_state:
                    request_data["session_id"] = st.session_state.session_id
                    
                # Send request to API with a timeout
                response = requests.post(
                    f"{API_URL}/chat",
                    json=request_data,
                    timeout=10  # Set a timeout to avoid hanging indefinitely
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Save the session ID
                    st.session_state.session_id = response_data["session_id"]
                    
                    # Display the response
                    assistant_response = response_data["response"]
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to the backend API. Please make sure the backend server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
            except requests.exceptions.Timeout:
                error_msg = "Request timed out. The backend server might be overloaded or not responding."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
            except Exception as e:
                error_msg = f"Error communicating with the API: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
