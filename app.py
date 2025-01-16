__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
import os
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFaceEndpoint
import sqlite3
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

os.environ['LANGCHAIN_API_KEY']="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"
os.environ['LANGCHAIN_PROJECT']="GenAIAPPWithOPENAI"
os.environ['HF_TOKEN']="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"
os.environ['LANGCHAIN_TRACING_V2']="true"

st.set_page_config(page_title="Medical Chatbot", page_icon=':robot:')
st.title("Medical Chatbot")
LOCALDB='USE_LOCALDB'

radio_opt=["Use SQLLITE 3 DB- medical.db","Use Medical ChatBot"]
select_opt=st.sidebar.radio(label='select the Agent you want to use',options=radio_opt)
hf_key=os.environ['HF_TOKEN']
repo_id='openai-community/gpt2'
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7, token=hf_key)

def get_session_history(session_id:str)->BaseChatMessageHistory:
  if session_id not in st.session_state.store:
    st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

with st.spinner('Loading...'):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
      model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
      )

    
    dataset = load_dataset("RafaelMPereira/HealthCareMagic-100k-Chat-Format-en")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)

    docs=text_splitter.create_documents(dataset['train'][('text')])
    vectore_store=Chroma.from_documents(documents=docs, embedding=embeddings)

    retriever=vectore_store.as_retriever()


    
    st.success('Loading is done!')


if radio_opt.index(select_opt)==0:
  LOCALDB='USE_LOCALDB'
  db_uri=LOCALDB
  st.cache_resource(ttl='2h')
  def config_db(db_uri):
    dbfilepath=(Path('/content/medical.db')).absolute()
    creator=lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
    return SQLDatabase(create_engine("sqlite:///",creator=creator))

  db= config_db(db_uri)
  toolkit=SQLDatabaseToolkit(db=db,llm=llm)
  agent=create_sql_agent(llm=llm,
                       toolkit=toolkit,
                       verbose=True,
                       agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
  if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
  for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
  user_query=st.chat_input(placeholder="Ask anything from the database")

  if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

elif radio_opt.index(select_opt)==1:
  contextualize_q_system_prompt=(
              "Given a chat history and the latest user question"
              "which might reference context in the chat history, "
              "formulate a standalone question which can be understood "
              "without the chat history. Do NOT answer the question, "
              "just reformulate it if needed and otherwise return it as is."
          )
  contextualize_q_promt=ChatPromptTemplate.from_messages([
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ])
  history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_promt)

  system_prompt=(
                "You are a highly knowledgeable and reliable AI-powered healthcare assistant, specifically designed to assist users with their medical"
                " queries. Your goal is to provide accurate, concise, and helpful information based on retrieved medical knowledge and database entries."
                " Here are your responsibilities:"
                " Understand Context: Accurately comprehend the user's question, including medical terms, symptoms, or treatment-related queries."
                " Retrieve Information: Use the provided medical database and embeddings to fetch the most relevant and precise information."
                " Augment Responses: If the retrieved information is insufficient, supplement your response with API data or generate an accurate answer based on your training."
                " Provide Clear Responses: Ensure your answers are user-friendly, avoiding overly technical language while maintaining accuracy."
                " Admit Gaps: If no relevant information is available, admit it and direct the user to consult a healthcare professional."
                " Example Behavior:"
                " Input: 'What are the symptoms of diabetes?'"
                " Response: 'The common symptoms of diabetes include excessive thirst, frequent urination, and fatigue. For further details, consult a doctor.'"
                " Input: 'How to treat fever in children?'"
                " Response: 'Fever in children can be treated with rest, hydration, and fever-reducing medications like acetaminophen. Contact a pediatrician if the fever persists for more than 3 days.'"
                " Input: 'Can you provide the latest treatment for asthma?'"
                " Response: 'The latest treatments for asthma include biologics like omalizumab for severe cases. Please consult your doctor for personalized advice.'"
                " Your response must be:"
                " Fact-based and well-researched."
                " Supported by retrieved documents or supplemented by API outputs."
                " Contextually appropriate and tailored to the user's query."
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
  qa_propmt=ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ])

  question_answer_chain=create_stuff_documents_chain(llm,qa_propmt)

  rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
  session_id="default_session"
  if "store" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.store={}
  coversation_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_message_key="input",
            history_message_key="chat_history",
            output_message_key="answer"
        )
  user_input=st.text_input("your question")
  if user_input:
    session_history=get_session_history(session_id)
    response=coversation_rag_chain.invoke({"input",user_input},
                                              config={"configurable":{"session_id",session_id}})
    st.write("Assistant",response['answer'])
    st.write(st.session_state.store)
    st.write("Chat History:", session_history.messages)



