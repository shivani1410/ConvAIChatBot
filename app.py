__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from pathlib import Path
import os
import sqlite3
from datasets import load_dataset
from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
# from langchain.agents.langchain_community import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
import sqlitecloud

ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ["ASTRA_DB_API_ENDPOINT"] = ASTRA_DB_API_ENDPOINT
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = ASTRA_DB_APPLICATION_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
api_key=os.environ['GROQ_API_KEY']

st.set_page_config(page_title="Medical Chatbot", page_icon=':robot:')
st.title("Medical Chatbot")

llm =ChatGroq(groq_api_key=api_key,model_name='Llama3-8b-8192',streaming=True)
session_id=st.text_input("Enter Your Name (without space)")
if session_id:
  st.write('Model is Loading')
if "messages" not in st.session_state or 'store' not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.store={}
### Q&A Chatbot
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vstore = AstraDBVectorStore(
    collection_name="medical",
    embedding=embeddings,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
if vstore.astra_env.collection.estimated_document_count()==0:
  dataset = load_dataset("RafaelMPereira/HealthCareMagic-100k-Chat-Format-en")['train'].select(range(500))

  docs = []
  for entry in dataset:
      doc = Document(page_content=entry['text'])
      docs.append(doc)

  inserted_ids = vstore.add_documents(docs)

retriever = vstore.as_retriever(search_kwargs={"k": 3})
contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


def get_session_history(session:str)->BaseChatMessageHistory:
  if session_id not in st.session_state.store:
    st.session_state.store[session_id]=ChatMessageHistory()
  if not st.session_state.store[session_id].messages:
    st.session_state.store[session_id].add_ai_message("How can I help you?")
  return st.session_state.store[session_id]

conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

#### Database query chabot

st.cache_resource(ttl='2h')
# def config_db():
#   return SQLDatabase(create_engine("sqlitecloud://cylddamonk.g1.sqlite.cloud:8860/chinook.sqlite?apikey=Hgal3hl0QpFMyED2Y3Z0XBNsdFiKLnN2kQGM1Y0qZJQ"))


# db= config_db()

# toolkit=SQLDatabaseToolkit(db=db,llm=llm)
# agent=create_sql_agent(llm=llm,
#                        toolkit=toolkit,
#                        verbose=True,
#                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if session_id:
  st.write('Model is Loaded')
  user_input = st.text_input("Your question:")
  if user_input:
    if any(keyword in user_input.lower() for keyword in ['end', 'quit', 'finish', 'bye', 'goodbye']):
      st.write('Thank you for the conversation')
      st.session_state.store={}
      st.session_state.messages={}

    if any(keyword in user_input.lower() for keyword in ['table', 'select', 'insert', 'update', 'delete','schema','db','database']):
      for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
      st.session_state.messages.append({"role": "user", "content": user_input})
      with st.chat_message("assistant"):

        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_input,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

    else:
      response = conversational_rag_chain.invoke(
                  {"input": user_input},
                  config={
                      "configurable": {"session_id":session_id}
                  },
              )
      session_history=get_session_history(session_id)
      st.write("Assistant:", response['answer'])
      with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
      with st.expander("Chat History"):
        for i,doc in enumerate(session_history.messages):
          # st.write(type(doc))
          # if doc['type']=='human':
          if doc.type=='ai':
           
            st.write('Bot:'+doc.content)
          else:
            st.write('You:'+doc.content)
          st.write('------------------------')

else:
    st.warning("Please enter your Name")



