# Import necessary libraries
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import torch
import os, operator
from langchain.agents import Tool
from typing import TypedDict, Annotated
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import GoogleSerperResults
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

# Suppress unnecessary warnings
set_verbosity_error()

# Load environment variables from .env file
load_dotenv(find_dotenv())
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Initialize the Google search tool
search_tool = GoogleSerperResults(max_results=2)

# Define available tools list
tools = [
    search_tool
]

# Define the state structure for the agent
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # List of messages with add operator for combining

# Initialize the language model from Hugging Face
model = HuggingFaceEndpoint(task="text-generation", 
                            repo_id="meta-llama/Llama-3.2-3B-Instruct", 
                            )

# Create a chat interface for the model
llm = ChatHuggingFace(llm=model, verbose=True)

# Define the Agent class to handle conversations and tool usage
class Agent:
    def __init__(self, model, tools, system=""):
        """
        Initialize the agent with a model, tools, and optional system prompt.
        
        Args:
            model: The language model to use
            tools: List of tools the agent can use
            system: System prompt to guide the agent's behavior
        """
        self.system = system
        
        # Create a state graph for managing conversation flow
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)  # Node for LLM processing
        graph.add_node("action", self.call_tool)  # Node for tool execution
        
        # Define conditional edges in the graph
        graph.add_conditional_edges(
            "llm",
            self.take_action,
            {True:"action", False:END},  # If action needed, go to action node, else end
        )
        graph.add_edge("action", "llm")  # After action, return to LLM
        graph.set_entry_point("llm")  # Start with LLM node
        
        # Set up memory for conversation persistence
        memory = MemorySaver()
        self.graph = graph.compile(checkpointer=memory)
        
        # Create tool dictionary for easy lookup
        self.tools = {t.name: t for t in tools}
        
        # Bind tools to the model
        self.model = model.bind_tools(tools)

    def take_action(self, state: AgentState):
        """
        Determine if a tool should be called based on the model's response.
        
        Args:
            state: Current conversation state
            
        Returns:
            Boolean indicating whether to call a tool
        """
        tool = state["messages"][-1]
        return len(tool.tool_calls) > 0

    def call_model(self, state: AgentState):
        """
        Process the current state with the language model.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with model's response
        """
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def call_tool(self, state: AgentState):
        """
        Execute tool calls requested by the model.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with tool execution results
        """
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# Define the system prompt for the agent
prompt = """You are a smart research assistant responsible for generating most accurate answers to the question asked. \
Use the search engine to look up information!
"""

# Initialize the agent with the model, tools, and system prompt
abot = Agent(llm, tools, system=prompt)

# Specify a thread ID for conversation tracking
config = {"configurable": {"thread_id": "4"}}

# Example 1: Ask a question about world population
messages = [HumanMessage(content="Which is the most populated country in the world? Can you tell me more about that country?", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)

print("---------------- Summary ----------------------")
# Example 2: Ask for a summary of the previous response
messages = [HumanMessage(content="Now summarize that response", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)