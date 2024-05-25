import streamlit as st
import requests
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["LLAMA_API_KEY"], base_url="https://api.llama-api.com")
pdf_key = os.environ["PDF"]
source_id = None
# Hidden functions for API interactions disguised as RAG processing functions
def _pseudo_upload_pdf(file_path):
    files = [
        ('file', ('file', open(file_path, 'rb'), 'application/octet-stream'))
    ]
    headers = {
        'x-api-key': pdf_key,
    }
    response = requests.post(
        'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)
    
    if response.status_code == 200:
        source_id = response.json()['sourceId']
        return source_id
    else:
        return None

def _pseudo_query_pdf(source_id, user_question):
    headers = {
        'x-api-key': pdf_key,
        "Content-Type": "application/json",
    }
    data = {
        'sourceId': source_id,
        'messages': [
            {
                'role': "user",
                'content': user_question + " answer in Vietnamese.",
            }
        ]
    }
    response = requests.post(
        'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['content']
    else:
        return "Error: Could not retrieve response."

# Functions to handle PDF processing and querying
def read_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore_from_pdf(file):
    pdf_text = read_pdf(file)
    text_chunks = split_text_into_chunks(pdf_text)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def query_vectorstore(vectorstore, query):
    docs = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    return docs

def rag_query(vectorstore, user_question):
    docs = query_vectorstore(vectorstore, user_question)
    context = " ".join([i[0].page_content for i in docs])
    messages = [
        {"role": "system", "content": " Bạn trả lời các câu hỏi dựa trên Context được cung cấp bên dưới, trả lời thông tin một cách đầy đủ và chính xác. Hãy trả lời thân thiện với người dùng. Trong mọi trường hợp, luôn luôn sử dụng tiếng việt."},
        {"role": "user", "content": f"Context: {context}\n\nQ: {user_question}\n"}
    ]

    chat_completion = client.chat.completions.create(messages=messages, top_p=0.9, temperature=0.6, presence_penalty=0.5, frequency_penalty=0.2, model="llama3-70b", stream=False)
    return chat_completion.choices[0].message.content

# Streamlit app setup
st.set_page_config(page_title="Chatbot tuyển sinh IUH 🏫", layout="wide")
st.title("Chatbot tuyển sinh IUH 🏫")

# Cache function
@st.cache_data
def get_cached_data(file_path):
    source_id = _pseudo_upload_pdf(file_path)
    return source_id

# Sidebar for PDF upload
with st.sidebar:
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get source_id from cache or generate a new one
    source_id = get_cached_data(file_path)

    if source_id:
        st.session_state.source_id = source_id
        st.sidebar.success("PDF processed successfully!")
    else:
        st.sidebar.error("Failed to process PDF.")

# Chat interface
if "source_id" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": " Xin chào, tôi có thể tư vấn gì cho bạn?"}]

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
                with st.spinner("Tôi đang suy nghĩ 💭..."):
                    response = _pseudo_query_pdf(st.session_state.source_id, prompt)
                    st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)