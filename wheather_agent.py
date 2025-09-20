from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv  
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
checkpointer=InMemorySaver()

class WeatherResponse(BaseModel):
    conditions: str

# Define the tool first
def get_current_weather(city: str) -> str:
    '''Get the current weather in a given city.'''
    return f"The weather of the city {city} is 25C and sunny."

# Initialize model
model = init_chat_model(
    model_provider="groq",
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Create the agent
agent = create_react_agent(
    model=model,
    tools=[get_current_weather],
     checkpointer=checkpointer,
     response_format=WeatherResponse
)

config={"configurable":{"thread_id":"1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print(response["messages"])
        break
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    
    print(response["structured_response"])
    ai_message = response["messages"][-1]
    ai_content = ai_message.content
    print(ai_content)  # Print the AI's response
