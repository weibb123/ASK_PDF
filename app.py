# bring in streamlit for UI/app interface
import streamlit as st
from streamlit_chat import message
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import os
from langchain.memory import ConversationBufferMemory
import openai
from helper import *

def set_api_key(api_key: str):
    try:
        st.session_state['OPENAI_API_KEY'] = api_key
    except Exception as e:
        st.error("wrong api key?")

def create_vectorstore(text_chunks):
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
        
    except:
        return st.error("did I get API key?")
    
    
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# user submit pdf -> get_pdf from user -> get text chunks from pdf -> create vector database -> retrieve based on similarit
def create_chain(vectorstore):
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
    except Exception as e:
        return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
        
    llm = ChatOpenAI(openai_api_key=API_KEY,
                     temperature=0, model='gpt-3.5-turbo-1106')
    

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    con_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return con_chain

def create_chat(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for index, message in enumerate(st.session_state.chat_history):
        if index % 2 == 0: 
            st.write(message.content)

        else: # LLMs answer lies in odd number index
            st.write(message.content)

def main():

    # page setup
    st.set_page_config(page_title="😊AI reads PDF")

    # Initialization
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    # header
    st.header("AI reader")
    # Set up option
    
    query = st.text_input("Ask question about your documents:")
    if query:
        create_chat(query)
        
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader("upload a pdf file", accept_multiple_files=True)
        API_KEY = st.text_input("OpenAI API Key",
                                            placeholder="Paste your OpenAI API key here (sk-...)",
                                            type="password")
        
        if st.button("Process pdf"):
            if API_KEY:
                set_api_key(API_KEY)
                
                with st.spinner("Processing"):
                    # get pdf text
                    text = get_pdf(uploaded_file)
                    st.write("Finish getting pdf")

                    # get text chunks
                    text_chunks = get_chunks(text)
                    st.write("Finish getting chunks")

                    # create vector database
                    vectorstore = create_vectorstore(text_chunks)
                    st.write("Finish creating vector database")

                    # conversation chain keeps track of whole history + conversation
                    st.session_state.conversation = create_chain(vectorstore)
                    st.write("Finish creating chains")
           

if __name__ == '__main__':
    main()

