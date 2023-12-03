import streamlit as st
import langchain
import torch
from io import BytesIO
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
import re


def get_pdf(doc):
    text = ""

    for pdf in doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    """use it after using get_pdf function"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#def parse_pdf(file):
    pdf = pypdf.PdfReader(file)
    pages = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        pages.append(text)
    
    # convert into document
    page_docs = [Document(page_content=page) for page in pages]

    # add page numbers as metadata
    for index, doc in enumerate(page_docs):
        doc.metadata["page"] = index + 1  # remember index starts at 0, page starts at 1
    
    # split pages into chunks
    final_doc = []

    for doc in page_docs:
        text_splitter =  RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        docs = text_splitter.split_text(doc.page_content)
        for i, doc in enumerate(docs):
            docu = Document(
                page_content=docs, metadata={"page": doc.metadata["page"], "doc": i}
            )
            # add sources
            docu.metadata["source"] = f"{docu.metadata['page']}-{docu.metadata['doc']}"
            final_doc.append(docu)
    
    return final_doc


def process_pdf(file):
    """Expect input from extract_pdf..
    For example: process_pdf(create_pdf(filename))"""
    # load documents
 
    # split documents
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(file)
    
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    vectorstore = Chroma.from_documents(docs, embeddings)

    return vectorstore