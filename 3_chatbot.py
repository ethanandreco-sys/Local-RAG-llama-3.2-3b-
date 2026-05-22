import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local Text RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Raw Text (100% Local RAG)")

# Initialize components once safely using standard local paths

# @st.cache_resource
def initialize_rag():
    db_path = "./chroma_db"
    collection_name = "local_rag_collection"
    # ... rest of your code remains exactly the same
    
    # 1. Local Embeddings Setup (No ngrok, targets localhost by default)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. Local Vector Database Connection
    if os.path.exists(db_path):
        vector_store = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings,
            collection_name=collection_name
        )
        retriever_obj = vector_store.as_retriever(search_kwargs={"k": 3})
    else:
        st.sidebar.error("⚠️ Database directory not found! Run your ingestion script first.")
        retriever_obj = None
        
    # 3. Local Chat Model Setup (No ngrok, targets localhost by default)
    llm_obj = ChatOllama(
        model="llama3.2:3b",
        temperature=0.2
    )
    
    return retriever_obj, llm_obj

retriever, llm = initialize_rag()

# Format helper for context documents
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)

# Define clean system prompt template
system_prompt = (
    "You are an expert assistant. Answer the user's question using exclusively the provided context. "
    "If you do not know the answer based on the context, state that you do not know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# Build Chain
if retriever is not None:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None

# Process active chat conversations
if user_query := st.chat_input("Ask a question about your input text:"):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if rag_chain is None:
            st.error("RAG chain is uninitialized because the local database files are missing.")
        else:
            with st.spinner("Analyzing text chunks and generating response on your Mac Mini..."):
                try:
                    response_text = rag_chain.invoke(user_query)
                    st.markdown(response_text)
                    
                    # Pull and display source materials for transparency
                    source_docs = retriever.invoke(user_query)
                    with st.expander("See Referenced Context Chunks"):
                        for idx, doc in enumerate(source_docs):
                            st.caption(f"Chunk {idx + 1}:")
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")