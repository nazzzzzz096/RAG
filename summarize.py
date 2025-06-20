from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

def load_local_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1000
    )
    return HuggingFacePipeline(pipeline=pipe)

def summarize_pdf(pdf_path: str, model_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Use local flan-t5-small
    llm = load_local_llm(model_path)

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)

    print("SUMMARY:\n", summary)

if __name__ == "__main__":
    summarize_pdf("C:\week31\project\data\case.pdf", "C:\\week31\\legal-doc-ai-assistant\\flan-t5-small")
