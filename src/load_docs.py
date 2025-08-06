import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(doc_dir="data/documents"):
    docs = []
    for file in os.listdir(doc_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(doc_dir, file))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
