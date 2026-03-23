from langgraph.graph import StateGraph, START, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
    

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    message = state['messages']

    response = model.invoke(message)

    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


## Streaming example

# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# for message_chunk,metadata in chatbot.stream(
#     {'messages': [HumanMessage(content="Write an essay about cricket in 100 words.")]}, 
#     config=CONFIG,
#     stream_mode = 'messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end='|', flush=True)

