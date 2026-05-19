import streamlit as st
from langchain_chroma import Chroma
#  Modern imports that natively support the 'headers' parameter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local Text RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Raw Text (Local RAG)")

# Update ONLY the initialize_rag() block inside 3_chatbot.py:
@st.cache_resource
def initialize_rag():
    # ⚡ Use your exact active ngrok forwarding link
    public_ollama_url = "https://ngrok-free.dev" 
    
    # ⚡ FIXED: Added client_kwargs to inject the browser warning bypass header
    bypass_headers = {"client_kwargs": {"headers": {"ngrok-skip-browser-warning": "true"}}}
    
    # Route embedding math to your Mac mini over the web with bypass headers
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=public_ollama_url,
        **bypass_headers
    )
    
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="local_rag_collection"
    )
    
    # Route LLM text generation to your Mac mini over the web with bypass headers
    llm = ChatOllama(
        model="llama3.2:3b", 
        temperature=0.2,
        base_url=public_ollama_url,
        **bypass_headers
    )
    return vector_store.as_retriever(search_kwargs={"k": 3}), llm


retriever, llm = initialize_rag()

# Format helper for context documents
def format_docs(docs):
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

# Create RAG pipeline using clean, modern LCEL syntax
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}

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
            # FIXED: Uses invoke() instead of get_relevant_documents()
            source_docs = retriever.invoke(user_query)
            
            # Execute the modern LCEL RAG chain
            response_text = rag_chain.invoke(user_query)
            st.markdown(response_text)
            
            # Display source grounding snippets
            with st.expander("See Referenced Context Chunks"):
                for idx, doc in enumerate(source_docs):
                    st.caption(f"Chunk {idx + 1}:")
                    st.write(doc.page_content)
