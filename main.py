import os
import pandas as pd
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["open_ai_credentials"].key

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

def load_llm(model: str, settings: dict):
    if model == 'OpenAI': return OpenAI(**settings)

def load_web_data(collection: str, site: str, embeddings: object, llm: object):
    docs = WebBaseLoader(site).load()
    context = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(docs)
    vec_db = Chroma.from_documents(context, embeddings, collection_name=collection)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vec_db.as_retriever())
    return chain

def load_agent_prompt(tools: list):
    prefix = """Have a conversation with a human, \
        answering the following questions as best you can. \
        Kindly reject any queries when response is not within given context. \
        You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    return ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix,\
        input_variables=["input", "chat_history", "agent_scratchpad"])

def load_agent(llm: object, tools: list):
    llm_chain = LLMChain(llm=llm, prompt=load_agent_prompt(tools))
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    memory = ConversationBufferMemory(memory_key="chat_history")
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main_frame, langchain_settings = st.columns([2, 1])

    with st.sidebar:
        llm_model = st.selectbox("Select a LLM Model", ['OpenAI'])    # TODO: Add LLM model setting
        st.info(f"{llm_model} Model Settings")
        settings = {
            'temperature': st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7),
            'max_tokens': st.slider('Maximum Length', min_value=128, max_value=1028, value=256),
            'top_p': st.slider('Top P', min_value=0.0, max_value=1.0, value=1.0),
            'frequency_penalty' : st.slider('Frequency penalty', min_value=0.0, max_value=2.0, value=0.0),
            'presence_penalty': st.slider('Presence penalty', min_value=0.0, max_value=2.0, value=0.0)
        }

    llm = load_llm(llm_model, settings)

    if 'tools' not in st.session_state: st.session_state.tools = list()
    if 'desc' not in st.session_state: st.session_state.desc = dict()

    with langchain_settings:
        with st.form("New Chain Form"):
            chain_name = st.text_input("Key").lower()
            link = st.text_input("Webpage/site link")
            desc = st.text_area("Use Case")
            embeddings = OpenAIEmbeddings()

            if st.form_submit_button("Create"): 
                chain = load_web_data(chain_name, link, embeddings, llm)
                st.session_state.desc[chain_name] = {'Link': link, 'Description': desc}
                st.session_state.tools.append(Tool(name=chain_name, func=chain.run, description=desc))
        
        desc = pd.DataFrame.from_dict(st.session_state.desc).T
        st.dataframe(desc, use_container_width=True)

    with main_frame:    # TODO: Add Demo gif, Establish chat playground
        demo, playground = st.tabs(["Demo", "Playground"])
        with playground:
            chat_area = st.container()
            if st.session_state.tools:
                agent = load_agent(llm, st.session_state.tools)
                st.info("Assistant Agent Initiated")
                with chat_area: message("How may I assist you today?") 
                with st.form("User Input"):
                    query = st.text_area("User:")
                    if st.form_submit_button("Query"):
                        with chat_area: 
                            message(query, is_user=True)
                            message(agent.run(query))
            else: st.warning("No Keys Initialized, unable to establish Assistant")