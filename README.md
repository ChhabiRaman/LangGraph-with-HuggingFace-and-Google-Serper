# LangGraph Chatbot

A conversational AI assistant with tool usage capabilities, built using LangGraph and LangChain.

## 📋 Overview

This project implements a conversational agent that can use external tools (like search engines) to provide informative responses to user queries. It leverages a graph-based approach for conversation management, allowing the agent to:

1. Process user messages
2. Determine when to use tools to gather information
3. Execute tools and process their results
4. Maintain conversation context across interactions

## ✨ Features

- 🧠 Uses Llama-3.2-3B-Instruct for natural language understanding
- 🔍 Integrates with Google Search (via Serper API) for information retrieval
- 📊 Graph-based state management for complex conversation flows
- 🧩 Modular architecture that can be extended with additional tools
- 💾 Conversation memory to maintain context

## 🛠️ Requirements

- Python 3.9 or higher
- Required packages:
  - `transformers`
  - `langchain`
  - `langgraph`
  - `torch`
  - `python-dotenv`

## 🚀 Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/langgraph_chatbot.git
   cd <PROJECT_DIRECTORY>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   ```

## 📝 Usage

The main script demonstrates how to use the chatbot:

```python
from main import abot

# Set up a configuration with a thread ID for conversation tracking
config = {"configurable": {"thread_id": "your_thread_id"}}

# Create a message
messages = [HumanMessage(content="What is the capital of France?", name="user")]

# Get a response
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)
```

## 🏗️ Project Structure

```
langgraph_chatbot/
├── main.py          # Main implementation of the chatbot
├── .env             # Environment variables (API keys)
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## 📊 How It Works

The chatbot uses a state graph architecture from LangGraph:

1. The conversation state consists of messages exchanged between the user and the bot
2. The LLM node processes the current state and generates a response
3. A conditional edge determines whether to call a tool or end the conversation
4. If a tool is needed, the Action node executes the tool and adds its result to the state
5. The conversation flows back to the LLM to process the tool's result

## 💡 Example

```python
# Ask about the most populated country
messages = [HumanMessage(content="Which is the most populated country in the world?", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)

# Ask for a summary
messages = [HumanMessage(content="Now summarize that response", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)
```