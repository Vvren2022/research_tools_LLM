import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings

# Set API Key

os.environ["OPENAI_API_KEY"] = grisha_key

st.title("GrishaBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
vectorstore_openai = None  # Ensure variable is initialized globally

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if vectorstore_openai is not None:  # Ensure vectorstore is available
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.error("Please process URLs first before asking a question.")
