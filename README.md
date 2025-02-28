# 🤖 Llama 3 AI Chatbot

![Chatbot Interface](https://raw.githubusercontent.com/username/llama3-chatbot/main/docs/images/chatbot-banner.png)

A lightweight, user-friendly chatbot application powered by Meta's Llama 3 model via AWS Bedrock. This application consists of a FastAPI backend and a Streamlit frontend for a seamless chat experience.

## Features

- 💬 Interactive chat interface with conversation history
- 🧠 Powered by Meta's Llama 3 (8B parameter instructional model)
- 🔄 Session management with conversation persistence
- 🌐 RESTful API for easy integration
- 📱 Responsive web interface built with Streamlit

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- AWS CLI configured with appropriate permissions

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/llama3-chatbot.git
   cd llama3-chatbot
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
   python back.py
   ```

2. In a separate terminal, start the frontend:
   ```
   python front.py
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
llama3-chatbot/
├── back.py         # FastAPI backend server
├── front.py        # Streamlit frontend application
├── requirements.txt # Project dependencies
└── .env            # Environment variables (create this file)
```

## Dependencies

- FastAPI - Backend API framework
- Streamlit - Frontend web application
- LangChain - LLM integration framework
- AWS Bedrock - AI model provider

## License

MIT

## Acknowledgements

- This project uses Meta's Llama 3 model via AWS Bedrock
- Built with LangChain for seamless AI integration
