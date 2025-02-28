# README.md
"""
# AI Chatbot with LangChain, Bedrock, FastAPI, and Streamlit

This project implements a conversational AI chatbot using:
- **Amazon Bedrock** with the Llama 3 model for generating responses
- **LangChain** for conversation management and prompt engineering
- **FastAPI** as the backend API
- **Streamlit** for the frontend user interface

## Setup Instructions

### Prerequisites
- Python 3.9+
- AWS CLI configured with access to Amazon Bedrock
- AWS credentials with permissions for Bedrock

### Installation

1. Clone this repository
2. (Optional) Create a `.env` file in the `backend` directory to customize settings:
```
AWS_REGION=your_aws_region  # Optional: will use AWS CLI default if not specified
MODEL_ID=meta.llama3-70b-instruct-v1
TEMPERATURE=0.7
MAX_TOKENS=500
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd frontend
pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
cd backend
python chatbot_backend.py
```

2. Start the Streamlit frontend:
```bash
cd frontend
streamlit run chatbot_frontend.py
```

3. Open your browser and go to `http://localhost:8501` to use the chatbot.

## Features
- Uses AWS CLI for authentication (no need to specify credentials)
- Persistent conversation memory within sessions
- Customizable model parameters
- Clean and intuitive user interface
- Session management for multiple conversations

## Configuration
You can customize the application by modifying the environment variables in the `.env` file or directly in `config.py`.
"""
