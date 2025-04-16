import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

st.title("📊 Table Extractor + LLM Q&A + CSV/Excel Generator")

# URL input
url = st.text_input("🔗 Enter URL with HTML Tables", "")

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)

def extract_tables(url):
    res = requests.get(url)
    tables = pd.read_html(res.text)
    return tables

# Download functionality
def get_download_link(df, file_format="csv"):
    from io import BytesIO, StringIO
    if file_format == "csv":
        csv = df.to_csv(index=False)
        return csv.encode('utf-8')
    elif file_format == "excel":
        towrite = BytesIO()
        df.to_excel(towrite, index=False, engine='openpyxl')
        return towrite.getvalue()

# File Format Selector
file_format = st.selectbox("📁 Choose file format to download", ["csv", "excel"])

if url:
    st.info("ℹ️ You must run `python ingest_url.py <url>` once before asking questions.")
    question = st.text_input("❓ Ask a question about the tables:")

    if question:
        with st.spinner("Thinking..."):
            vectorstore = get_vectorstore()
            llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore)
            response = qa_chain.run(question)
            st.success("Answer:")
            st.write(response)

    st.divider()
    tables = extract_tables(url)
    for i, df in enumerate(tables):
        st.subheader(f"📋 Table {i+1}")
        st.dataframe(df, use_container_width=True)
        filename = f"table_{i+1}.{file_format}"
        st.download_button(
            label=f"📥 Download Table {i+1} as {file_format.upper()}",
            data=get_download_link(df, file_format),
            file_name=filename,
            mime="text/csv" if file_format == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
