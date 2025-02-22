# AI-Driven Knowledge Retrieval and Enhanced Question-Answering System

![](https://github.com/sujikathir/Ai-driven-Chatbot-for-enhanced-question-answering-system/blob/main/source/cover%20page.jpg)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Key Components](#key-components)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
6. [Generative AI Concepts](#generative-ai-concepts)
7. [Amazon Bedrock](#amazon-bedrock)
8. [Model Parameters](#model-parameters)
9. [Chatbot Architecture](#chatbot-architecture)
10. [Future Improvements](#future-improvements)

## Project Overview

This project implements an advanced AI-driven knowledge retrieval and question-answering system. It leverages state-of-the-art language models and innovative techniques to provide accurate, context-aware responses to user queries. The system is designed to understand and generate human-like text, making it suitable for a wide range of applications, from customer support to research assistance.

## Technologies Used

- Hugging Face Transformers
- AWS Bedrock
- LangChain
- Streamlit

## Key Components

1. **Customized Foundation Models**: Utilizes Hugging Face Transformers for various NLP tasks including text generation, summarization, and question-answering.
2. **AWS Bedrock Integration**: Leverages AWS Bedrock APIs and SDKs to enhance language models with advanced capabilities.
3. **LangChain Implementation**: Incorporates LangChain to add memory capabilities to the language models, enabling more coherent and context-aware conversations.
4. **Retrieval Augmented Generation (RAG)**: Implements RAG approach by fine-tuning pre-trained models on domain-specific data, improving the relevance and accuracy of generated responses.

## System Architecture

The system is built on a modular architecture that allows for easy integration of different components:

1. **Foundation Models Layer**: Customized language models based on Hugging Face Transformers.
2. **Enhancement Layer**: AWS Bedrock and LangChain integrations for improved model capabilities.
3. **Retrieval Layer**: Implementation of RAG for accurate information retrieval.
4. **User Interface**: Streamlit-based frontend for user interaction.

## Implementation Details

1. **Model Customization**: 
   - Fine-tuned Hugging Face Transformer models for specific tasks (text generation, summarization, QA).
   - Utilized domain-specific data to improve model performance in targeted areas.

2. **AWS Bedrock Integration**:
   - Implemented AWS Bedrock APIs and SDKs to access advanced AI capabilities.
   - Utilized Amazon Titan and other foundation models available through Bedrock.

3. **LangChain Implementation**:
   - Incorporated LangChain to add memory capabilities to the models.
   - Enabled the system to maintain context across multiple interactions.

4. **RAG Implementation**:
   - Fine-tuned pre-trained models on domain-specific data.
   - Implemented retrieval mechanisms to augment model responses with relevant information.

## Generative AI Concepts

- **Generative AI (GenAI)**: AI capable of creating various types of data (images, videos, audio, text, 3D models).
- **Machine Learning**: Broader field involving algorithms that learn patterns from data for predictions or decisions.
- **Large Language Models (LLMs)**: Advanced systems trained on vast amounts of text data to generate human-like language.

## Amazon Bedrock

- Fully managed service offering access to leading foundation models through a single API.
- Allows experimentation with various foundation models for different use cases.
- Supports private customization of models using techniques like fine-tuning and RAG.
- Serverless architecture eliminates the need for infrastructure management.

## Model Parameters

- **Temperature**: Controls randomness in generated text. Higher values increase diversity but may reduce coherence.
- **Top-p**: Limits the pool of tokens considered during text generation, focusing on the most probable ones.
- **Max_gen_len**: Specifies the maximum length of generated text.

## Chatbot Architecture

1. **User Interface**: Streamlit-based frontend for user input and displaying responses.
2. **Question Processing**: Analyzes and processes user questions.
3. **Prompt Generation**: Combines past messages with the current question to create a context-aware prompt.
4. **Model Interaction**: Sends the prompt to Amazon Bedrock for processing.
5. **Response Parsing**: Interprets and formats the model's output.
6. **Display**: Shows the answer in the Streamlit interface.
7. **History Management**: Uses LangChain for reading and writing conversation history.

## Installation and Setup

### Prerequisites
- Python installed
- Visual Studio Code installed
- AWS account with necessary permissions for Amazon Bedrock

### Steps

1. **Create Project Directory**
   - Open Visual Studio Code
   - Create a new folder for your project

2. **Set Up Virtual Environment**
   Open a terminal in VS Code and run:
   ```bash
   python -m venv venv

**Activate the virtual environment:**

- Windows: ``` venv\Scripts\activate ```
- macOS/Linux: ``` source venv/bin/activate ```

3. **Install Required Packages**
With the virtual environment activated, run:

```bash
pip install langchain boto3 streamlit
```

4. **Create Python Files**
In your project folder, create two files:

- chatbot_backend.py
- chatbot_frontend.py

Copy the provided code into these files respectively.

5. **Configure AWS Credentials**
Ensure your AWS credentials are set up, either through:

- AWS CLI configuration
- Setting environment variables

**Running the Application**
In the terminal, navigate to your project directory and run:

```bash
streamlit run chatbot_frontend.py
```

The chatbot should now be accessible through a web browser. Streamlit will provide a local URL (typically http://localhost:8501) where you can interact with your chatbot.

**Troubleshooting**

If you encounter issues:

- Check that all imports are resolved
- Verify your AWS credentials are correctly set up
- Ensure you have the necessary permissions for Amazon Bedrock

## Future Improvements

1. Implement more advanced RAG techniques for improved information retrieval.
2. Explore multi-modal capabilities to handle image and audio inputs.
3. Enhance the user interface for better user experience.
4. Implement more robust error handling and edge case management.
5. Optimize model performance for faster response times.
