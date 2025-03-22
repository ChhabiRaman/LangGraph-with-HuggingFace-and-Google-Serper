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
# from langchain_community.tools.google_serper import GoogleSerperResults
from langchain_community.tools import GoogleSerperResults
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

set_verbosity_error()
load_dotenv(find_dotenv())
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search_tool = GoogleSerperResults(max_results=2)

tools = [
    search_tool
]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

model = HuggingFaceEndpoint(task="text-generation", 
                            repo_id="meta-llama/Llama-3.2-3B-Instruct", 
                            )

llm = ChatHuggingFace(llm=model, verbose=True)

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.call_tool)
        graph.add_conditional_edges(
            "llm",
            self.take_action,
            {True:"action", False:END},
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        memory = MemorySaver()
        self.graph = graph.compile(checkpointer=memory)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def take_action(self, state: AgentState):
        tool = state["messages"][-1]
        return len(tool.tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def call_tool(self, state: AgentState):
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

prompt = """You are a smart research assistant responsible for generating most accurate answers to the question asked. \
Use the search engine to look up information!
"""

abot = Agent(llm, tools, system=prompt)

#specify a thread
config = {"configurable": {"thread_id": "4"}}
messages = [HumanMessage(content="Which is the most populated country in the world? Can you tell me more about that country?", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)

print("---------------- Summary ----------------------")
messages = [HumanMessage(content="Now summarize that response", name="user")]
result = abot.graph.invoke({"messages": messages}, config)
print(result['messages'][-1].content)
