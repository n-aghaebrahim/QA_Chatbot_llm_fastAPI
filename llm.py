import openai
import uvicorn
from fastapi import FastAPI, Request, Form
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch



import os
import datetime

#import IPython.display
from PIL import Image
import base64
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import tkinter as tk

from langchain.llms import OpenAI
from langchain.document_loaders import (
    DataFrameLoader,
    TextLoader,
    PyPDFLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import (
    DocArrayInMemorySearch,
    Chroma
)
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


#from dotenv import load_dotenv, find_dotenv
#from dotenv import load_dotenv, find_dotenv




# Set your OpenAI API key here
def set_openai_api_key(api_key):
    openai.api_key = api_key





# function to convert data and load it into panda format
# load data and preprocess it
def squad_json_to_dataframe(file_path, record_path=['data','paragraphs','qas','answers']):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    """

    file = json.loads(open(file_path).read())
    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file,record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    data = m[['id','question','context','answers']].set_index('id').reset_index()
    data['c_id'] = data['context'].factorize()[0]
    return data



def preprocess(data):
    data['answers'] = data['answers'].apply(lambda x: x[0]['text'] if x else None)

    # create a new data structure combine questions and answers
    # add $ at then end so its going to be easier to chunking later
    data['qa'] = data['question'] +data['answers']+'$'

    return data

def data_loader(data):
    # load the dataframe into loader
    # context
    loader = DataFrameLoader(data, page_content_column="qa")
    doc = loader.load()
    doc = doc[:1000]
    return doc


def create_text_splits(doc): 
    # splitting text into the specific chunck sizes
    # defining the overlap size for each chunck

    #from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator = "$",
        chunk_size = 125,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )


    splits = text_splitter.split_documents(doc)
    return splits

def initialize_openai_embeddings():
    embedding = OpenAIEmbeddings(request_timeout=60)
    return embedding

def get_gpt_model():
    # get the specific gpt model
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    print(llm_name)
    return llm_name


def create_docarray_in_memory_search(data, embedding): 
    db = DocArrayInMemorySearch.from_documents(data, embedding)
    return db

def create_vectordb(splits, embedding):
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
    )


    # EXAMPLES:
    #question = "What are major topics for this class?"

    #docs = vectordb.similarity_search(question,k=4)
    #print(docs[0].metadata['answers'])
    return vectordb



def initialize_llm_chatbot(llm_name, temperature=0):
    # create chatbot
    llm = ChatOpenAI(model_name=llm_name, temperature=temperature)
    
    # define chatbot memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return llm, memory

def create_prompt_template(input_variables):
    # Build prompt
    template = """
    start by greeting to the Stanfor chatbot.\n
    try to ask the user Name, and remember it and when you respons back say the user Name as well.\n
    Also, try to memorize the converstation, and act like you are a human and responding.\n
	You are like an QA agent that you suppose to answer the question that you know.\n
	You will always gretting every one at the beging, also you can ask for their name so you will respond back with their name to be more polit.\n
	Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.\n
	Also, if you answered any question except being greedy you can say something like "Do you have any other question that I can help with?".\n
	If the person says I don't have any furthur questions, just say something like: I am always here to help you with any questions that you may have.\n
	{context}\n

	Question: {question}
	Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=input_variables,template=template,)
    return QA_CHAIN_PROMPT 


def initialize_qa_chain(llm, vectordb, QA_CHAIN_PROMPT):
    # Run chain

    #retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Examples
    # test topics
    #question = "Is probability a class topic?"

    #result = qa_chain({"query": question})
    #result["result"]
    return qa_chain, qa



# Set your OpenAI API key here
set_openai_api_key("sk-xxxxxxxxxxxxx")
data_df = squad_json_to_dataframe("data/train-v1.1.json") # convert json to dataframe
data_df = preprocess(data_df)
data_loader = data_loader(data_df)
splits = create_text_splits(data_loader)
embedding = initialize_openai_embeddings()
llm_name = get_gpt_model()
db = create_docarray_in_memory_search(data_loader, embedding)
vectordb = create_vectordb(splits, embedding)
llm, memory = initialize_llm_chatbot(llm_name, temperature=0)
QA_CHAIN_PROMPT = create_prompt_template(["context", "question"])
qa_chain, qa = initialize_qa_chain(llm, vectordb, QA_CHAIN_PROMPT)


def get_bot_response(user_message):
    result = qa({"question": user_message})
    response = result["answer"]
    #result = qa_chain({"query": user_message})
    #response = result["result"]

    return str(response)






