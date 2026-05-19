import os
import streamlit as st
import requests
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

st.set_page_config(page_title="Local Text RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Raw Text (Local RAG)")

# ⚡ DIRECT PROXY EMBEDDING DRIVER
class StableNgrokEmbeddings(Embeddings):
    def __init__(self, model="nomic-embed-text", base_url="", headers=None):
        self.model = model
        self.url = f"{base_url}/api/embed"
        self.headers = headers or {}

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                payload = {"model": self.model, "input": text}
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
                response.raise_for_status()
                embeddings.append(response.json()["embeddings"])
            except Exception:
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text):
        try:
            payload = {"model": self.model, "input": text}
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception:
            return [0.0] * 768

# ⚡ DIRECT PROXY LLM CHAT DRIVER 
class StableNgrokChatModel(BaseChatModel):
    model: str = "llama3.2:3b"
    base_url: str = ""
    headers: dict = {}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        url = f"{self.base_url}/api/chat"
        
        formatted_messages = []
        for msg in messages:
            role = "user" if msg.type == "human" else "system" if msg.type == "system" else "assistant"
            formatted_messages.append({"role": role, "content": msg.content})
            
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        
        response = requests.post(url, json=payload, headers=self.headers, timeout=120)
        response.raise_for_status()
        
        text_output = response.json()["message"]["content"]
        generation = ChatGeneration(message=AIMessage(content=text_output))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "stable_ngrok_chat"

# Initialize components once
@st.cache_resource
def initialize_rag():
    # 🚨 REPLACE: Paste your exact active ngrok forwarding domain endpoint link here
    public_ollama_url = "https://ngrok-free.dev" 
    ngrok_headers = {"ngrok-skip-browser-warning": "true"}
    
    embeddings = StableNgrokEmbeddings(
        model="nomic-embed-text",
        base_url=public_ollama_url,
        headers=ngrok_headers
    )
    
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
    except Exception:
        retriever_obj = None
        
    llm_obj = StableNgrokChatModel(
        model="llama3.2:3b",
        base_url=public_ollama_url,
        headers=ngrok_headers
    )
    return retriever_obj, llm_obj

retriever, llm = initialize_rag()

# Format helper for context documents
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)

# Define system prompt template
system_prompt = (
    "You are an expert assistant. Answer the user's question using exclusively the provided context. "
    "If you do not know the answer based on the context, state that you do not know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

if retriever is not None:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}

        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = (
        {"context": lambda x: "No context available.", "question": RunnablePassthrough()}

        | prompt
        | llm
        | StrOutputParser()
    )

# Process active chat conversations
if user_query := st.chat_input("Ask a question about your input text:"):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing text chunks and generating response on your Mac Mini..."):
            response_text = rag_chain.invoke(user_query)
            st.markdown(response_text)
            
            if retriever is not None:
                try:
                    source_docs = retriever.invoke(user_query)
                    with st.expander("See Referenced Context Chunks"):
                        for idx, doc in enumerate(source_docs):
                            st.caption(f"Chunk {idx + 1}:")
                            st.write(doc.page_content)
                except Exception:
                    pass
