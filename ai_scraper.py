import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Crawler", 
    layout="wide",
    page_icon="🕸️"
)

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re
from urllib.parse import urlparse

# Prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer logical and concise.
Question: {question}
Context: {context}
Answer:
"""

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'indexed_url' not in st.session_state:
    st.session_state.indexed_url = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set up model and embeddings
@st.cache_resource
def get_model_and_embeddings():
    embeddings = OllamaEmbeddings(model="llama3.2")
    model = OllamaLLM(model="llama3.2")
    return embeddings, model

embeddings, model = get_model_and_embeddings()

# Validate URL
def validate_url(url):
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}(?:\.\d{1,3}){3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$',  # path
        re.IGNORECASE
    )
    if not url_pattern.match(url):
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and parsed.netloc
    except:
        return False

# Helper functions
def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    return loader.load()

def split_text(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(documents)

def index_docs(documents):
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    return vector_store

def retrieve_docs(query, vector_store, k=3):
    return vector_store.similarity_search(query, k=k)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# UI
st.title("🕸️ AI Crawler with LangChain & Ollama")
st.markdown("Ask questions about any web page content using AI-powered retrieval and generation.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
    num_results = st.slider("Retrieved Documents", 1, 10, 3)
    
    if st.session_state.indexed_url:
        st.success(f"✅ Indexed: {st.session_state.indexed_url}")
        if st.button("Clear Index"):
            st.session_state.vector_store = None
            st.session_state.indexed_url = None
            st.session_state.chat_history = []
            st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    url = st.text_input("Enter a web page URL to crawl:", placeholder="https://example.com")
    
    if url and url != st.session_state.indexed_url:
        if not validate_url(url):
            st.error("❌ Please enter a valid HTTP/HTTPS URL")
        else:
            if st.button("🔍 Crawl and Index Page", type="primary"):
                try:
                    with st.spinner("Loading and indexing the page..."):
                        documents = load_page(url)
                        if not documents:
                            st.warning("⚠️ No documents found at the URL.")
                        else:
                            chunks = split_text(documents, chunk_size, chunk_overlap)
                            st.session_state.vector_store = index_docs(chunks)
                            st.session_state.indexed_url = url
                            st.session_state.chat_history = []
                            
                            st.success(f"✅ Indexed {len(chunks)} chunks!")
                            st.info(f"📄 Total content length: {sum(len(doc.page_content) for doc in documents)} characters")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    if st.session_state.indexed_url:
        st.markdown("### 📊 Index Stats")
        st.metric("Status", "Ready ✅")
        st.metric("URL", st.session_state.indexed_url.split('/')[2])

# Chat interface
st.markdown("---")
st.markdown("### 💬 Ask Questions")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("🔍 View Retrieved Context"):
                st.write(message["context"])

if st.session_state.vector_store:
    question = st.chat_input("Ask a question about the page...")
    
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.spinner("🤔 Thinking..."):
            try:
                docs = retrieve_docs(question, st.session_state.vector_store, num_results)
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = answer_question(question, context)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context
                })
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("🔍 View Retrieved Context"):
                        st.write(context)
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg
                })
else:
    st.info("👆 Please crawl a web page first to start asking questions!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    Built with Streamlit, LangChain, and Ollama • 
    Make sure Ollama is running with llama3.2 model installed
    </div>
    """, 
    unsafe_allow_html=True
)
