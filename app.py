import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import tempfile

load_dotenv()
grok_api_key = os.getenv('GROQ_API_KEY')

if not grok_api_key:
    st.error("No GROQ_API_KEY found. Please check your .env file.")
    st.stop()

llm = ChatGroq(api_key=grok_api_key, model='gemma2-9b-it')

prompt = ChatPromptTemplate.from_template('''
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
''')

def create_vector_embeddings(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        final_docs = text_splitter.split_documents(all_docs)

        st.session_state.vectors = FAISS.from_documents(
            final_docs, st.session_state.embeddings
        )
        st.success("Vector database is ready!")

# PDF Upload UI
uploaded_pdfs = st.file_uploader(
    "Upload your PDF research papers", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_pdfs and st.button("Process PDFs"):
    create_vector_embeddings(uploaded_pdfs)

user_prompt = st.text_input('Enter your query from the uploaded PDFs')

if user_prompt and "vectors" in st.session_state:
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)

    response = retrieval_chain.invoke({'input': user_prompt})

    st.write(response.get('answer', 'No answer found.'))

    with st.expander('Document similarity search'):
        context_docs = response.get('context', [])
        for i, doc in enumerate(context_docs):
            st.write(doc.page_content)
            st.write('-----------------------------------')
