from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def create_vectorstore(docs, persist_dir="./chroma_db"):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb
