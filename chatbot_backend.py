"""
Simplified FastAPI backend for the AI chatbot with AWS Bedrock and LangChain integration
"""
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_HOST = "127.0.0.1"  # Using explicit IP instead of localhost
API_PORT = 8080  # Using different port in case 8000 is in use

# AWS Region - can be set in .env or will use default from AWS CLI config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1 if not set

# LLM Configuration
MODEL_ID = "meta.llama3-8b-instruct-v1:0"  # AWS Bedrock model ID
TEMPERATURE = 0.7  # Controls randomness in the model's responses
MAX_TOKENS = 500  # Maximum number of tokens to generate

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
    session_id: str = None
    message: str

class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    response: str
    history: List[ChatMessage]

# Create LangChain components
try:
    from langchain_aws import BedrockLLM
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    
    def create_llm_chain():
        """Create a LangChain conversation chain"""
        # Initialize the Bedrock LLM
        llm = BedrockLLM(
            model_id=MODEL_ID,
            region_name=AWS_REGION,
            model_kwargs={
                "temperature": TEMPERATURE,
                "max_gen_len": MAX_TOKENS,
                "prompt": "",  # Placeholder for the prompt
            }
        )
        
        # Create a conversation memory
        memory = ConversationBufferMemory(return_messages=True)
        
        # Define template
        template = """
        <chat>
        <system>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        </system>
        
        {history}
        <user>{input}</user>
        <assistant>
        </assistant>
        </chat>
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
    
    # Store conversation chains
    conversation_chains = {}
    
except ImportError:
    print("Warning: LangChain or AWS packages not found. Running in mock mode.")
    # Mock implementation for testing without AWS dependencies
    conversation_chains = {}
    
    def create_llm_chain():
        """Mock chain that just returns a placeholder response"""
        from collections import namedtuple
        
        MockChain = namedtuple('MockChain', ['memory', 'predict'])
        MockMemory = namedtuple('MockMemory', ['chat_memory'])
        MockMessages = namedtuple('MockMessages', ['messages'])
        
        class MockMessage:
            def __init__(self, role, content):
                self.type = role
                self.content = content
        
        # Create a mock memory with placeholder messages
        mock_messages = [
            MockMessage("human", "Test message"),
            MockMessage("ai", "Test response")
        ]
        
        mock_chat_memory = MockMessages(mock_messages)
        mock_memory = MockMemory(mock_chat_memory)
        
        # Create a predict function that returns a placeholder
        def mock_predict(input):
            mock_messages.append(MockMessage("human", input))
            response = f"This is a mock response to: {input}"
            mock_messages.append(MockMessage("ai", response))
            return response
        
        # Return the mock chain
        return MockChain(memory=mock_memory, predict=mock_predict)

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"status": "API is running", "endpoints": ["/chat", "/sessions/{session_id}"]}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response"""
    # Create or retrieve session
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation_chains[session_id] = create_llm_chain()
    elif session_id not in conversation_chains:
        conversation_chains[session_id] = create_llm_chain()
    
    # Get the conversation chain for this session
    conversation_chain = conversation_chains[session_id]
    
    try:
        # Format the input correctly for the Bedrock model
        formatted_input = {
            "prompt": request.message  # Ensure the input is sent as a "prompt"
        }
        
        # Get response from the model
        response = conversation_chain.predict(input=formatted_input)
        
        # Extract conversation history
        history = []
        for message in conversation_chain.memory.chat_memory.messages:
            history.append(ChatMessage(
                role="user" if message.type == "human" else "assistant",
                content=message.content
            ))
        
        return ChatResponse(
            session_id=session_id,
            response=response,
            history=history
        )
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in conversation_chains:
        del conversation_chains[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8080")
    uvicorn.run("back:app", host="127.0.0.1", port=8080, reload=True)
