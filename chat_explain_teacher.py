import streamlit as st
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from llamaapi import LlamaAPI
from openai import OpenAI
import os

load_dotenv()
# Initialize LlamaAPI and OpenAI clients
client = OpenAI(api_key=os.environ["LLAMA_API_KEY"], base_url="https://api.llama-api.com")

def read_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_from_pdf(file):
    pdf_text = read_pdf(file)
    text_chunks = split_text_into_chunks(pdf_text)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def query_vectorstore(vectorstore, query):
    docs = vectorstore.similarity_search_with_relevance_scores(query, k=4)
    return docs

def rag_query(vectorstore, user_question):
    docs = query_vectorstore(vectorstore, user_question)
    context = " ".join([i[0].page_content for i in docs])
    messages = [
        {"role": "system", "content": "Bạn trả lời các câu hỏi dựa trên Context được cung cấp bên dưới, trả lời thông tin một cách đầy đủ và chính xác. Hãy trả lời thân thiện với người dùng. Trong mọi trường hợp, luôn luôn sử dụng tiếng việt."},
        {"role": "user", "content": f"Context: {context}\n\nQ: {user_question}\n"}
    ]

    chat_completion = client.chat.completions.create(messages=messages, top_p=0.9, temperature=0.6, presence_penalty=0.5, frequency_penalty=0.2, model="llama3-70b", stream=False)
    return chat_completion.choices[0].message.content

# Streamlit app setup
st.set_page_config(page_title="PDF Chatbot with RAG", layout="wide")
st.title("PDF Chatbot with RAG")

# Sidebar for PDF upload
with st.sidebar:
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    

if uploaded_file is not None and "vectorstore" not in st.session_state:
    
    vectorstore = get_vectorstore_from_pdf(uploaded_file)
    st.session_state.vectorstore = vectorstore
    st.sidebar.success("PDF processed successfully!")

# Chat interface
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input("Ask your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_query(st.session_state.vectorstore, prompt)
                    st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
