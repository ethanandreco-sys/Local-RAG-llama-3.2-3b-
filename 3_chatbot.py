import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Local Text RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Raw Text (Local RAG)")

@st.cache_resource
def initialize_rag():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="local_rag_collection"
    )
    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
    return vector_store.as_retriever(search_kwargs={"k": 3}), llm

retriever, llm = initialize_rag()

system_prompt = (
    "You are an expert assistant. Answer the user's question using exclusively the provided context. "
    "If you do not know the answer based on the context, state that you do not know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if user_query := st.chat_input("Ask a question about your input text:"):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing text chunks..."):
            response = rag_chain.invoke({"input": user_query})
            st.markdown(response["answer"])
            
            with st.expander("See Referenced Context Chunks"):
                for idx, doc in enumerate(response["context"]):
                    st.caption(f"Chunk {idx + 1}:")
                    st.write(doc.page_content)
