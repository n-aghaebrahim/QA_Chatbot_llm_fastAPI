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
OPENAI_API_KEY = "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"


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

# load QA dataset
filename = "train-v1.1.json"
data = squad_json_to_dataframe(filename)

data['answers'] = data['answers'].apply(lambda x: x[0]['text'] if x else None)


# create a new data structure combine questions and answers
# add $ at then end so its going to be easier to chunking later
data['qa'] = data['question'] +data['answers']+'$'

# load the dataframe into loader
# context
loader = DataFrameLoader(data, page_content_column="qa")
doc = loader.load()


doc = doc[:1000]


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


embedding = OpenAIEmbeddings(request_timeout=60)


# get the specific gpt model
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)


db = DocArrayInMemorySearch.from_documents(doc, embedding)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
)



question = "What are major topics for this class?"

docs = vectordb.similarity_search(question,k=4)
print(docs[0].metadata['answers'])



# create chatbot
llm = ChatOpenAI(model_name=llm_name, temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Build prompt
template = """
You are like an QA agent that you suppose to answer the question that you know.\n
You will always gretting every one at the beging, also you can ask for their name so you will respond back with their name to be more polit.\n
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.\n
Also, if you answered any question except being greedy you can say something like "Do you have any other question that I can help with?".\n
If the person says I don't have any furthur questions, just say something like: I am always here to help you with any questions that you may have.\n
{context}\n

Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain

#retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})




# test topics
question = "Is probability a class topic?"

result = qa_chain({"query": question})
result["result"]


# define memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# chatbot model initialization
#vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)


agent = qa

def get_bot_response(user_message):
    result = qa_chain({"query": user_message})
    response = result["result"]

    return str(response)









