import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query_tool, list_tables, describe_tables_tool
import os
from dotenv import load_dotenv
    
load_dotenv()
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
MODEL_VERSION = os.environ["OPENAI_CHAT_MODEL"]
EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]

st.set_page_config(page_title="Gen AI based Invoice Entity Extraction Bot", page_icon = "images/favicon.png", layout = 'wide', initial_sidebar_state = 'auto')
    
# Add logo to the sidebar
with st.sidebar:
    st.image("images/logo_light.png", width=150)

# Add logout menu to the sidebar
with st.sidebar:
    st.text("Welcome : Admin")
    st.button("Logout")

st.title("Gen AI - SQL Database Bot")

chat = AzureChatOpenAI(
    deployment_name=MODEL_VERSION,
    openai_api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    temperature=0.1,
    verbose=True,
)

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [
    run_query_tool,
    describe_tables_tool
]

agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
    memory=memory
)

## Session State Variable
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['human']}, {'answer':message['AI']})
  
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
       
st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )        

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("How can I assist you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    output = agent_executor(prompt)
    message = {'human':prompt, 'AI':output["output"]}
    st.session_state.chat_history.append(message)
    # print(":::::::::::::",st.session_state.chat_history)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
 
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(output["output"])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output["output"]})
    # print("------------",st.session_state.messages)