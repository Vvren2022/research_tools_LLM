import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import dill as pickle


os.environ["OPENAI_API_KEY"] = grisha_key

st.title("GrishaBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    raw_data = loader.load()  # raw text data from URLs

    # Ensure raw data is in a format that can be converted into Document objects
    docs = []
    for text in raw_data:
        # Make sure the page_content is a valid string
        page_content = str(text)  # Ensure it's a string
        docs.append(Document(page_content=page_content))

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    split_docs = text_splitter.split_documents(docs)  # Split Document objects

    # Create embeddings and save them
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([doc.page_content for doc in split_docs])  # Get embeddings

    # Save embeddings to a file (instead of FAISS index)
    with open(file_path, "wb") as f:
        pickle.dump(vectors, f)
    main_placeholder.text("Embeddings Saved...✅✅✅")
    time.sleep(2)

    # Add download button for the saved FAISS file
    with open(file_path, "rb") as f:
        st.download_button(
            label="Download FAISS Store",
            data=f,
            file_name="faiss_store_openai.pkl",
            mime="application/octet-stream"
        )

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectors = pickle.load(f)  # Load the embeddings (vectors)

            # Rebuild FAISS index from the loaded vectors
            vectorstore_openai = FAISS.from_embeddings(vectors)  # Assuming `FAISS.from_embeddings` method is available
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
