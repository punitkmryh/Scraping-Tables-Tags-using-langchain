import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def extract_tables(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)
    table_data = []
    
    for idx, df in enumerate(tables):
        csv_data = df.to_csv(index=False)
        table_data.append(f"Table {idx+1}:\n{csv_data}")
    
    return table_data, tables

def create_vectorstore(table_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents(table_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore/faiss_index")

if __name__ == "__main__":
    url = sys.argv[1]
    table_data, _ = extract_tables(url)
    create_vectorstore(table_data)
    print("✅ Tables extracted and stored in vector DB.")
