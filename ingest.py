import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob

DATA_PATH = 'pdfs/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom DirectoryLoader to print filenames before loading
class CustomDirectoryLoader(DirectoryLoader):
    def lazy_load(self):
        for path in glob.glob(self.path + '/*.pdf'):
            print(f"Loading document: {path}")
            yield from PyPDFLoader(path).lazy_load()

# Create vector database
def create_vector_db():
    start_time = time.time()
    print("Start loading documents")
    loader = CustomDirectoryLoader(DATA_PATH)
    documents = list(loader.lazy_load())
    print(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    print(f"Created embeddings in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    db = FAISS.from_documents(texts, embeddings)
    print(f"Created FAISS index in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    db.save_local(DB_FAISS_PATH)
    print(f"Saved FAISS index in {time.time() - start_time:.2f} seconds")

    # Print the number of embeddings
    print(f"Number of embeddings: {len(texts)}")

# Load the vector database
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    print("Loading vector database with dangerous deserialization allowed")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Loaded vector database")
    return db

if __name__ == "__main__":
    create_vector_db()
