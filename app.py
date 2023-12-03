# bring in streamlit for UI/app interface
import streamlit as st
from dotenv import load_dotenv
import torch
from streamlit_chat import message
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from openai.error import OpenAIError
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
import openai
from helper import *

# user submit pdf -> get_pdf from user -> get text chunks from pdf -> create vector database -> retrieve based on similarit
def create_chain(vectorstore):
        
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":512})

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
    torch.cuda.empty_cache()
    load_dotenv()
    # page setup
    st.set_page_config(page_title="😊AI reads PDF")

    # container for chat history
    response_container = st.container()

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
        if st.button("Process pdf"):
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

