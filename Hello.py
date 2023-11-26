import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import time
from langchain.docstore.document import Document
import fitz


def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # documents = [uploaded_file.read().decode()]
        
        # loader = PyPDFLoader(uploaded_file.name) 
        # documents = loader.load()
        # documents = [documents[i].page_content for i in range(len(documents))]
        # documents = []
        # reader = PdfReader(uploaded_file.read())
        # i = 1
        # for page in reader.pages:
        #     documents.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        #     i += 1
        # print(documents)
        if uploaded_file is not None:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            documents = []
            for page in doc:
                # print(dir(page))
                documents.append(page.get_text())
            # st.write(text) 
            doc.close()
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='Cochlear Smart QA Engine')
st.title('Cochlear Smart QA Engine')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# print(dir(uploaded_file))
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    openai_api_key = "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"
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)