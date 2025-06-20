# 🧠 Legal Document AI Assistant

This project provides two core functionalities powered by local language models and LangChain:

1. **Retrieval-Augmented Generation (RAG)** — Ask questions and get answers based on a legal PDF.
2. **Summarization** — Generate concise summaries of legal documents.

---

## 📁 Project Structure
project/
│
├── app/
│ ├── ingest.py # Ingest PDF and create FAISS index
│ ├── rag_chat.py # Ask questions using RAG pipeline
│ ├── summarize.py # Summarize legal documents
│
├── vectorstore/
│ └── legal_index/ # FAISS index files (auto-generated)
│
├── data/
│ └── case.pdf # Sample legal PDF document
│
├── flan-t5-base/ # Local fine-tuned FLAN-T5 model (for RAG)
├── flan-t5-small/ # Local fine-tuned FLAN-T5 model (for summarization)
└── README.md # You're reading it!

🛠️ Technologies and Tools Used
🧠 LangChain
   Used for building the RAG pipeline and summarization logic:

   RetrievalQA for retrieval-augmented question answering

   load_summarize_chain for the map-reduce summarization strategy

   Prompt Template for custom prompt formatting

   RecursiveCharacterTextSplitter for splitting documents into chunks

   PyPDFLoader for reading PDF files

🤗 Hugging Face Transformers
   Used for loading and running local text-to-text models:

   AutoTokenizer, AutoModelForSeq2SeqLM: Load pre-trained T5 models (FLAN-T5 base/small)

   pipeline: Set up text2text-generation pipeline for both answering and summarizing

🧬 Hugging Face Sentence Transformers
  Used to generate dense vector embeddings for document chunks:

  Model: sentence-transformers/all-MiniLM-L6-v2 — small, fast, and efficient embedding model

📦 FAISS (Facebook AI Similarity Search)
  Used as the vector store backend to store and search over embeddings for document chunks:

  Enables efficient retrieval of top-k similar chunks based on a question

📂 Local Models
   All models are downloaded and run locally (no API calls or internet required):

   flan-t5-base for answering questions (QA)

   flan-t5-small for document summarization

🔄 Workflows & Concepts
  🔍 Retrieval-Augmented Generation (RAG)
     Combines retrieval (from FAISS vector store) and generation (via FLAN-T5) to produce accurate answers based on document context.

  📝 Summarization (Map-Reduce)
     Splits large documents, summarizes each chunk, and then combines them into a final summary using a map-reduce approach.

  🗂 Document Chunking
    Uses RecursiveCharacterTextSplitter to break PDFs into overlapping chunks for better context embedding and retrieval.

 🧪 Optional Parameters and Techniques
   k=5 in .as_retriever(k=5) retrieves the top 5 relevant chunks.

   max_new_tokens=256 or max_length=1000 limits the generation length.

  do_sample=False ensures deterministic outputs.

 allow_dangerous_deserialization=True is used for loading FAISS indexes from disk (be cautious).



This project demonstrates how to build a practical, local LLM-powered assistant for reading, querying, and summarizing legal documents. It’s fast, offline, and highly customizable. Ideal for both legal professionals and learners.
