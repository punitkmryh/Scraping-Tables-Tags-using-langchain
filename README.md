# Scraping-Table-Tags-WebPages-using-langchain

## Building a **LangChain-powered Table Extractor + LLM Q&A + Downloadable CSV/Excel Generator** deployed on **Streamlit Cloud**. 

Here's exactly what we’ll do:

---

### ✅ What This Project Does

1. 🔗 **Takes a user-specified URL** (with tables in HTML)
2. 🧠 Extracts all **HTML tables** using **BeautifulSoup + Pandas**
3. 💬 Chains the **tables with an LLM (via Groq)** for Q&A via RAG
4. 📂 Lets the user:
   - Ask specific queries about the table
   - Select table(s) to **download as CSV or Excel**
5. 🚀 Fully deployable on **Streamlit Cloud**
6. 💾 Uses **FAISS + HuggingFace Embeddings** for vector storage

---

## 📁 Project Structure

```
table-extractor-rag/
│
├── app.py                    # Streamlit app
├── ingest_url.py            # Fetches & chunks table text, saves FAISS
├── requirements.txt
├── .env                     # Contains GROQ API Key
├── vectorstore/             # Stores FAISS index (after ingestion)
│   └── faiss_index/
```

---

## 🔧 1. `ingest_url.py` – Scrape Tables & Create Vector DB

```python
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
```

---

## 🧠 2. `app.py` – Streamlit UI for Q&A + Downloads

```python
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
```

---

## 📦 3. `requirements.txt`

```txt
streamlit
python-dotenv
langchain
langchain-community
langchain-groq
beautifulsoup4
pandas
requests
faiss-cpu
sentence-transformers
openpyxl
```

---

## 🗝️ 4. `.env`

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Deployment Steps on Streamlit Cloud

1. **Run ingestion locally:**
   ```bash
   python ingest_url.py https://your-url-with-tables.com
   ```

2. ✅ Commit `vectorstore/faiss_index/` to GitHub.

3. 🆙 Push all files to a GitHub repo (e.g., `https://github.com/yourusername/table-rag-streamlit`)

4. Go to **Streamlit Cloud** → [https://streamlit.io/cloud](https://streamlit.io/cloud)

5. Create a new app:
   - Choose your GitHub repo
   - Set `app.py` as the main file

6. Add your **GROQ API Key** in **Secrets**:

   ```toml
   GROQ_API_KEY = "your-real-groq-api-key"
   ```

7. **Deploy!** 🎉

---

### ✅ Sample Test Link for Tables

You can try this with:
- [https://en.wikipedia.org/wiki/List_of_IITs](https://en.wikipedia.org/wiki/List_of_IITs)
- [https://en.wikipedia.org/wiki/Comparison_of_programming_languages](https://en.wikipedia.org/wiki/Comparison_of_programming_languages)
