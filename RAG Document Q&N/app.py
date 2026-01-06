import streamlit as st
from langchain_groq import ChatGroq

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


import os
from dotenv import load_dotenv
load_dotenv()

### load the GROQ API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model="Gemma-7b-It")
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    please provide the most accurate respone based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") #data ingetion
        st.session_state.docs=st.session_state.loader.load() # document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter you query from the research paper")
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({"input":user_prompt})
    print(f"response time:{time.process_time()-start}")

    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")