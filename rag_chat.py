from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate

def load_local_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_answer(query: str, index_path: str, model_path: str):
    # Load vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)

    # Use local FLAN-T5 model
    llm = load_local_llm(model_path)

    prompt_template = """Use the following context to answer the question. 
If the answer is not in the context, say "Answer not found".

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build retrieval-based QA
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(k=5),chain_type='stuff',chain_type_kwargs={'prompt':PROMPT})
    result = qa.invoke(query)
    answer = result["result"]
    if answer.strip() in ["[1]", "", "1"]:
        answer = "Answer not found"
    return {"result": answer}



if __name__ == "__main__":
    while True:
        q = input("Ask a legal question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = get_answer(q, "vectorstore/legal_index","C:\\week31\\project\\flan-t5-base")
        print("\nAnswer:", answer['result'])
