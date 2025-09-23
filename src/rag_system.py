import src.config as config
from src.embedding import EmbeddingFactory, EmbeddingOllama, EmbeddingGemini
from src.file_handler import load_files_from_directory, split_documents
import chromadb
import pandas as pd
import ollama

class RAGSystem:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embedding_method = None
        self.initialized = False
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.vector_store = self.client.get_or_create_collection(name="obsidian_notes")

    def initialize(self, limit=None):
        # Load documents
        self.documents = load_files_from_directory(config.FILE_VAULT)
        if limit:
            self.documents = self.documents[:limit]

        # Split documents into chunks
        self.chunks = pd.DataFrame(split_documents(self.documents))
        self.embedding_method = EmbeddingFactory.create(config.EMBEDDING_METHOD)

        # Create embeddings
        embeddings = [self.embedding_method.embed(chunk) for chunk in self.chunks["content"].tolist()]
        # add to collection
        self.vector_store.upsert(
            documents=self.chunks["content"].tolist(),
            metadatas=self.chunks.drop(columns=["content"]).to_dict(orient="records"),
            ids=[str(i) for i in range(len(self.chunks))],
            embeddings=embeddings
        )
        self.initialized = True
        return {"status": "success"}

    def get_documents_info(self):
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks)
        }

    def ask_question(self, question):
        if not self.initialized:
            raise Exception("RAG system not initialized")

        question_embedding = self.embedding_method.embed(question)
        relevant_chunks = self.vector_store.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        self.generate_answer(relevant_chunks, question)

        # display relevant chunks for debugging
        print("\nRelevant Chunks:") 
        for doc in relevant_chunks['documents'][0]:
            print(f"- {doc}")
        
        print("\nRelevant Chunks metadata:")
        for meta in relevant_chunks['metadatas'][0]:
            print(f"- {meta}")
        
        print("\nDistances:")
        for dist in relevant_chunks['distances'][0]:
            print(f"- {dist}")

    def generate_answer(self, chunks, question):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that helps people find information in a set of documents. Use the provided context to answer the question. If the context does not contain the answer, say you don't know."
            },
            {
                "role": "user",
                "content": f"Context: {chunks['documents'][0]}\n\nQuestion: {question}"
            },
        ]

        # Generate an answer from the relevant chunks
        for part in ollama.chat('mistral:latest', messages=messages, stream=True):
            print(part['message']['content'], end='', flush=True)