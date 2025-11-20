import os
import time
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

folder_path = "existential_texts"

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# File selection dropdown
selected_file = st.selectbox("Select a text file:", ["— Select a file —"] + txt_files)

# Creating vectorstore
@st.cache_resource
def load_vectorstore(file_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )

    with open(file_path, "r", encoding="utf-8") as f:
        doc = f.read()

    chunks = text_splitter.create_documents([doc])

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="existential_rag"
    )
    return vectordb

# Only build vectorstore after file selection
if selected_file != "— Select a file —":
    file_path = os.path.join(folder_path, selected_file)
    vectordb = load_vectorstore(file_path)
    retriever = vectordb.as_retriever(k=3)

    st.title(f"Existentialist Chatbot (RAG for '{selected_file}')")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            SystemMessage("Answer as an existentialist philosopher, using retrieved text when relevant.")
        )

    # show chat history
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat input
    prompt = st.chat_input("Ask a question...")

    if prompt:
        # save user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

        # retrieve context
        retrieved_docs = retriever.invoke(prompt)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])

        # append retrieval context before completion
        rag_prompt = f"Use the following context when helpful:\n\n{context_text}\n\nUser question: {prompt}"

        llm = ChatOllama(model="llama3.2:3b")

        result = llm.invoke([
            SystemMessage("Answer as an existentialist philosopher."),
            HumanMessage(rag_prompt)
        ]).content

        # show assistant reply
        with st.chat_message("assistant"):
            st.markdown(result)
        st.session_state.messages.append(AIMessage(result))
else:
    st.info("Please select a text file to enable the RAG chatbot.")
