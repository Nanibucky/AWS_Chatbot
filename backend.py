import os
import uuid
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8080

# AWS Region - can be set in .env or will use default from AWS CLI config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1 if not set

# LLM Configuration
MODEL_ID = "meta.llama3-70b-instruct-v1:0"  
TEMPERATURE = 0 
MAX_TOKENS = 500  

# Initialize FastAPI
app = FastAPI(title="AI Chatbot API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    response: str
    history: List[ChatMessage]

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    created_at: str
    last_active: str
    message_count: int

# Updated response extraction function to remove backticks
def extract_assistant_response(response):
    """
    Extract the clean response from the model output.
    """
    # Print the raw response for debugging
    print(f"Raw response from model: {response}")
    
    # Simple approach - if the response is clean, return it
    clean_response = response.strip()
    
    # If response has instruction tags, try to clean it
    if "[/INST]" in response:
        # Split by instruction tag and take the last part
        parts = response.split("[/INST]")
        if len(parts) > 1:
            clean_response = parts[-1].strip()
    
    # If there's a model token at the end, remove it
    clean_response = clean_response.replace("</s>", "").strip()
    
    # Remove triple backticks that might be surrounding the response
    clean_response = re.sub(r'^```.*?\n|```$', '', clean_response, flags=re.DOTALL)
    
    # Remove any other tags
    clean_response = re.sub(r'\[.*?\]', '', clean_response)
    
    # If the response is still empty or just contains tags
    if clean_response == "" or re.match(r'^\[/?[A-Za-z]+\]$', clean_response):
        # Fallback response
        clean_response = "I'm sorry, but I couldn't generate a proper response. Please try asking your question again."
    
    # Debug the cleaned response
    print(f"Cleaned response: {clean_response}")
    
    return clean_response

# Create LangChain components
try:
    from langchain_aws import BedrockLLM
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    
    def create_llm_chain():
        """Create a LangChain conversation chain with simplified prompt"""
        # Initialize the Bedrock LLM
        llm = BedrockLLM(
            model_id=MODEL_ID,
            region_name=AWS_REGION,
            model_kwargs={
                "temperature": TEMPERATURE,
                "max_gen_len": MAX_TOKENS
            }
        )
        
        # Create a conversation memory
        memory = ConversationBufferMemory()
        
        # Create a simple prompt template
        template = """
You are a helpful AI assistant. Please respond to the following question or request:

{input}

If you refer to our conversation history, here it is:
{history}

Please provide a direct response without adding prefixes like "Assistant:" or using any special formatting.
"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        # Create the conversation chain
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        
        return conversation_chain
    
    # Store conversation chains and metadata
    conversation_chains = {}
    session_metadata = {}
    
except ImportError:
    print("Warning: LangChain or AWS packages not found. Running in mock mode.")
    # Mock implementation for testing without AWS dependencies
    conversation_chains = {}
    session_metadata = {}
    
    def create_llm_chain():
        """Mock chain that just returns a placeholder response"""
        from collections import namedtuple
        
        MockChain = namedtuple('MockChain', ['memory', 'predict'])
        MockMemory = namedtuple('MockMemory', ['chat_memory', 'buffer'])
        
        class MockChatMemory:
            def __init__(self):
                self.messages = []
                
            def add_user_message(self, message):
                self.messages.append({"role": "user", "content": message})
                
            def add_ai_message(self, message):
                self.messages.append({"role": "assistant", "content": message})
        
        # Create mock components
        mock_chat_memory = MockChatMemory()
        mock_memory = MockMemory(mock_chat_memory, "")
        
        # Create a predict function that returns a placeholder
        def mock_predict(input, history=""):
            mock_chat_memory.add_user_message(input)
            response = f"This is a mock response to: {input}"
            mock_chat_memory.add_ai_message(response)
            return response
        
        # Return the mock chain
        return MockChain(memory=mock_memory, predict=mock_predict)

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {
        "status": "API is running", 
        "endpoints": ["/chat", "/sessions", "/sessions/{session_id}"],
        "active_sessions": len(conversation_chains)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response"""
    # Create or retrieve session
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation_chains[session_id] = create_llm_chain()
        session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0
        }
    elif session_id not in conversation_chains:
        conversation_chains[session_id] = create_llm_chain()
        session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0
        }
    
    # Update session metadata
    session_metadata[session_id]["last_active"] = datetime.now().isoformat()
    session_metadata[session_id]["message_count"] += 1
    
    # Get the conversation chain for this session
    conversation_chain = conversation_chains[session_id]
    
    try:
        # Get history string
        history = ""
        if hasattr(conversation_chain.memory, "buffer"):
            history = conversation_chain.memory.buffer

        # Print the input message for debugging
        print(f"Input message: {request.message}")
        
        # Get response from the model
        full_response = conversation_chain.predict(input=request.message, history=history)
        
        # Extract the clean response
        clean_response = extract_assistant_response(full_response)
        
        # Build history for the response
        history = []
        if hasattr(conversation_chain.memory, "chat_memory") and hasattr(conversation_chain.memory.chat_memory, "messages"):
            for message in conversation_chain.memory.chat_memory.messages:
                role = "user" if message.type == "human" else "assistant"
                content = message.content
                history.append(ChatMessage(role=role, content=content))
        
        return ChatResponse(
            session_id=session_id,
            response=clean_response,
            history=history
        )
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, metadata in session_metadata.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            created_at=metadata["created_at"],
            last_active=metadata["last_active"],
            message_count=metadata["message_count"]
        ))
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details about a specific chat session"""
    if session_id in conversation_chains and session_id in session_metadata:
        # Extract conversation history
        history = []
        if hasattr(conversation_chains[session_id].memory, "chat_memory") and hasattr(conversation_chains[session_id].memory.chat_memory, "messages"):
            for message in conversation_chains[session_id].memory.chat_memory.messages:
                role = "user" if message.type == "human" else "assistant"
                content = message.content
                history.append(ChatMessage(role=role, content=content))
        
        return {
            "session_id": session_id,
            "metadata": session_metadata[session_id],
            "history": history
        }
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in conversation_chains:
        del conversation_chains[session_id]
        if session_id in session_metadata:
            del session_metadata[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8080")
    uvicorn.run("back:app", host="127.0.0.1", port=8080, reload=True)
