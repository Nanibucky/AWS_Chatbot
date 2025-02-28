# 🤖 AI Chatbot

![Chatbot Interface](https://github.com/Nanibucky/AWS_Chatbot/blob/main/source/cover%20page.jpg)

A lightweight, user-friendly chatbot application powered by Foundation models via AWS Bedrock. This application consists of a FastAPI backend and a Streamlit frontend for a seamless chat experience. 

## Features

- 💬 Interactive chat interface with conversation history
- 🧠 Powered by Bedrock FM's
- 🔄 Advanced conversation management for coherent, context-aware dialogues 
- 🌐 FastAPI for easy integration
- 📱 Responsive web interface built with Streamlit

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- AWS CLI configured with appropriate permissions

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Nanibucky/AWS_Chatbot.git
   cd AWS_Chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your AWS configuration (optional):
   ```
   AWS_REGION=us-east-1
   ```

## Usage

1. Start the backend server:
   ```
   python backend.py
   ```

2. In a separate terminal, start the frontend:
   ```
   streamlit run frontend.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## API Endpoints

- `GET /` - Check API status
- `POST /chat` - Send a message and get a response
- `GET /sessions` - List all active sessions
- `GET /sessions/{session_id}` - Get details about a specific session
- `DELETE /sessions/{session_id}` - Delete a session

## Project Structure

```
AWS_Chatbot/
├── backend.py         # FastAPI backend server
├── frontend.py        # Streamlit frontend application
├── requirements.txt # Project dependencies
└── .env            # Environment variables (create this file)
```

## Dependencies

- FastAPI - Backend API framework
- Streamlit - Frontend web application
- LangChain - LLM integration framework
- AWS Bedrock - AI model provider

