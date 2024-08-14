from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

app = FastAPI()

class Query(BaseModel):
    question: str

class Response(BaseModel):
    result: str
    source_documents: list[str] = []

def set_custom_prompt():
    """ Prompt template for QA retrieval for each vectorstore """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    """ Load the locally downloaded model """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    try:
        print("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        print("Loading FAISS database...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Loading LLM...")
        llm = load_llm()
        print("Setting up QA chain...")
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        print("QA chain setup complete.")
        return qa
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

@app.post("/ask/")
async def ask(query: Query):
    print("Request received.")
    if not query.question:
        print("No question provided.")
        raise HTTPException(status_code=400, detail="Question is required")
    print(f"Question received: {query.question}")
    qa_result = qa_bot()
    if qa_result is None:
        print("Error loading vector database.")
        raise HTTPException(status_code=500, detail="Error loading vector database.")
    print("Processing question...")
    response = qa_result({'query': query.question})
    print("Question processed.")

    # Extract metadata
    source_documents = []
    for doc in response['source_documents']:
        source = doc.metadata.get('source', 'Unknown Document')
        page = doc.metadata.get('page', 'N/A')
        source_documents.append(f"{source} (Page {page})")
        print(f"Document metadata: {doc.metadata}")

    print("Response generated.")
    result_response = Response(result=response['result'], source_documents=source_documents)
    print(f"Returning response: {result_response}")

    return result_response

if __name__ == "__main__":
    import uvicorn

    print("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
