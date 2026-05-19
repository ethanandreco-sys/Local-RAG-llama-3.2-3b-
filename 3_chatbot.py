import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local Text RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Raw Text (Local RAG)")

# Initialize vector store and LLM once
@st.cache_resource
def initialize_rag():
    public_ollama_url = "https://ngrok-free.dev" 
    ollama_config = {"client_kwargs": {"headers": {"ngrok-skip-browser-warning": "true"}}}
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=public_ollama_url,
        **ollama_config
    )
    
    # SAFEGUARD: Automatically build directory paths if they are missing on the server
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        
    try:
        vector_store = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings,
            collection_name="local_rag_collection"
        )
        retriever_obj = vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.warning("⚠️ Database connection failed. Running in standalone model mode.")
        retriever_obj = None
        
    llm_obj = ChatOllama(
        model="llama3.2:3b", 
        temperature=0.2,
        base_url=public_ollama_url,
        **ollama_config
    )
    return retriever_obj, llm_obj

retriever, llm = initialize_rag()

# Format helper for context documents
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)

# Define structural system prompt
system_prompt = (
    "You are an expert assistant. Answer the user's question using exclusively the provided context. "
    "If you do not know the answer based on the context, state that you do not know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# FIXED: Check if retriever exists before building the LCEL pipeline chain
if retriever is not None:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}

        | prompt
        | llm
        | StrOutputParser()
    )
else:
    # Standalone model fallback chain if database context is unreachable
    rag_chain = (
        {"context": lambda x: "No context available.", "question": RunnablePassthrough()}

        | prompt
        | llm
        | StrOutputParser()
    )

# Handle user input via chat interface
if user_query := st.chat_input("Ask a question about your input text:"):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing text chunks..."):
            # Execute the modern LCEL RAG chain safely
            response_text = rag_chain.invoke(user_query)
            st.markdown(response_text)
            
            # Display source grounding snippets if retriever is active
            if retriever is not None:
                try:
                    source_docs = retriever.invoke(user_query)
                    with st.expander("See Referenced Context Chunks"):
                        for idx, doc in enumerate(source_docs):
                            st.caption(f"Chunk {idx + 1}:")
                            st.write(doc.page_content)
                except:
                    pass
