from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from  typing_extensions import  TypedDict
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.3
)

class AgentState(TypedDict):
    messages:Annotated[list, add_messages]

graph_build=StateGraph(AgentState)


def chatbot(state: AgentState) -> AgentState:
    response =llm.invoke(state["messages"])
    
    return { "messages":response}

graph_build.add_node("chatbot",chatbot)
graph_build.add_edge(START,"chatbot")
graph_build.add_edge("chatbot",END)

graph=graph_build.compile()

state=AgentState(messages=[])

while True:
    user_input=input("User:")
    if user_input.lower() in ["exit","quit","q"]:
        print(state["messages"])
        break
    state=graph.invoke({"messages":state["messages"]+[{"role":"user", "content":user_input}]})
    print(state["messages"][-1].content)


