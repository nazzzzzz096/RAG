from app.ingest import ingest_documents
from app.rag_chat import get_answer

# Step 1: Ingest
ingest_documents("C:\week31\project\data\case.pdf", "vectorstore/legal_index")

# Step 2: Ask
response = get_answer("who is the prime minister of india", "C:\week31\project\flan-t5-base")
print("Response:", response)
