from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next_state: str | None


class MessageClassifier(BaseModel):
    message_type: Literal["logical", "emotional"] = Field(
        ..., description="The type of the message, either 'logical' or 'emotional'.")


def message_classifier_agent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    classification = classifier_llm.invoke([
        {
            "role": "system",
            "content": "You are a message classifier. Classify the user's message as either 'logical' or 'emotional'."
        },
        {
            "role": "user",
            "content": f"Classify the following message: '{last_message.content}'"
        }
    ])

    print(f"Classified message type: {classification.message_type}")

    return {"message_type": classification.message_type}


def router(state: AgentState) -> AgentState:

    message_type = state.get("message_type", "logical")

    if message_type == "emotional":
        return {"next_state": "therapist"}

    return {"next_state": "logical"}


def logical_agent(state: AgentState) -> AgentState:

    last_message = state["messages"][-1]

    response = llm.invoke([
        {
            "role": "system",
            "content": "You are a logical assistant. Provide a logical and fact-based response to the user's message."
        },
        {
            "role": "user",
            "content": f"Respond logically to the following message: '{last_message.content}'"
        }
    ])
    return {"messages": [{"role": "assistant", "content": response.content}]}


def therapist_agent(state: AgentState) -> AgentState:

    last_message = state["messages"][-1]

    response = llm.invoke([
        {
            "role": "system",
            "content": "You are a compassionate therapist. Provide an empathetic and supportive response to the user's message."
        },
        {
            "role": "user",
            "content": f"Respond empathetically to the following message: '{last_message.content}'"
        }
    ])
    return {"messages": [{"role": "assistant", "content": response.content}]}


graph_build = StateGraph(AgentState)

graph_build.add_node("message_classifier", message_classifier_agent)
graph_build.add_node("router", router)
graph_build.add_node("logical", logical_agent)
graph_build.add_node("therapist", therapist_agent)

graph_build.add_edge(START, "message_classifier")
graph_build.add_edge("message_classifier", "router")

graph_build.add_conditional_edges(
    "router",
    lambda state: state.get("next_state"),
    {"logical": "logical", "therapist": "therapist"}
)

graph_build.add_edge("logical", END)
graph_build.add_edge("therapist", END)

graph = graph_build.compile()


def main():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input in ["exit", "quit"]:
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "multi_agent":
    main()
