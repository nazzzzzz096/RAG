from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_documents(pdf_path: str, index_save_path: str):
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Hugging Face Embeddings 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    vectorstore.save_local(index_save_path)
    print(f"Index saved to: {index_save_path}")

if __name__ == "__main__":
    ingest_documents("C:\week31\project\data\case.pdf", "vectorstore/legal_index")
