from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._last_docs = []

    def load_embeddings(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        print("Embeddings loaded!")

    def load_llm(self):
        print("Using semantic search for answers")

    def load_documents(self, pdf_path):
        print("Loading: " + pdf_path)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        print("Chunks: " + str(len(chunks)))
        return chunks

    def build_vectorstore(self, chunks):
        print("Building FAISS...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("Vector store ready!")

    def save_vectorstore(self, path="data/faiss_index"):
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path="data/faiss_index"):
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print("Vector store loaded!")

    def build_qa_chain(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("QA chain ready!")

    def answer(self, question):
        if not self.retriever:
            raise ValueError("Not initialized!")
        docs = self.retriever.invoke(question)
        self._last_docs = docs
        answer = docs[0].page_content if docs else "No content found."
        sources = [
            {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content[:200]}
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}
