"""
RAG System for document retrieval and question answering with advanced embedding
"""
import src.config as config
from src.embedding import create_embedding_manager, EmbeddingManager
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Optional
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
import langchain.chat_models.base
from dotenv import load_dotenv
import os

load_dotenv()


class RAGState(TypedDict):
    """State definition for the RAG application"""
    question: str
    context: List[Document]
    answer: str
    chroma_db: Chroma
    llm: langchain.chat_models.base.BaseChatModel


class RAGSystem:
    """RAG System for document retrieval and question answering with advanced embedding"""
    
    def __init__(self, use_advanced_embedding: bool = True, 
                 embedding_batch_size: int = 10,
                 embedding_delay: float = 1.0):
        """Initialize the RAG system with embeddings and LLM"""
        # Configuration d'embedding
        self.use_advanced_embedding = use_advanced_embedding
        
        if use_advanced_embedding:
            # Utiliser notre système d'embedding avancé
            self.embedding_manager = create_embedding_manager(
                provider="gemini",
                max_retries=3,
                batch_size=embedding_batch_size,
                delay_between_batches=embedding_delay
            )
            # Obtenir un wrapper compatible avec Chroma
            self.embeddings = self.embedding_manager.get_chroma_compatible_embedding()
        else:
            # Utiliser l'embedding standard LangChain
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            self.embedding_manager = None
        
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        self.vector_store = None
        self.documents = []
        self.chunks = []
        self.graph = None
        self.embedding_stats = {}
        
    def load_documents(self, directory_path: str = None, limit: int = None) -> List[Document]:
        """Load all .md files from a directory and return as a list of Documents."""
        if directory_path is None:
            directory_path = config.FILE_VAULT
        
        documents = []
        file_paths = list(Path(directory_path).rglob("*.md"))
        
        if limit:
            file_paths = file_paths[:limit]
            
        for file_path in file_paths:
            try:
                loader = TextLoader(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.documents = documents
        return documents
    
    def get_documents_info(self) -> dict:
        """Get information about loaded documents"""
        if not self.documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "file_paths": [],
                "status": "No documents loaded"
            }
        
        file_paths = [doc.metadata.get('source', 'Unknown') for doc in self.documents]
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "file_paths": file_paths,
            "status": "Documents loaded successfully"
        }
    
    def chunk_documents(self, chunk_size: int = 5000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.chunks = text_splitter.split_documents(self.documents)
        return self.chunks
    
    def setup_vector_store(self, progress_callback: Optional[callable] = None) -> None:
        """Setup the vector store with document chunks using advanced embedding if enabled"""
        if not self.chunks:
            raise ValueError("No chunks available. Call chunk_documents() first.")
        
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_DB_DIR,
        )
        
        # Utiliser l'embedding avancé si activé
        if self.use_advanced_embedding and self.embedding_manager:
            print("🚀 Utilisation du système d'embedding avancé...")
            
            # Embedder les documents avec notre système avancé
            embedding_result = self.embedding_manager.embed_documents(
                self.chunks, 
                progress_callback=progress_callback
            )
            
            # Stocker les statistiques
            self.embedding_stats = {
                "success_rate": embedding_result["success_rate"],
                "total_processing_time": embedding_result["total_processing_time"],
                "total_retries": embedding_result["total_retries"],
                "cache_hits": embedding_result["cache_hits"]
            }
            
            print(f"✅ Embedding terminé - Taux de succès: {embedding_result['success_rate']:.2%}")
            
            # Les embeddings sont automatiquement utilisés par le vector store via self.embeddings
            # On ajoute juste les documents pour que Chroma puisse les indexer
            self.vector_store.add_documents(documents=self.chunks)
        else:
            # Utiliser la méthode standard
            print("📝 Utilisation de l'embedding standard...")
            self.vector_store.add_documents(documents=self.chunks)
    
    def _retrieve(self, state: RAGState) -> dict:
        """Retrieve relevant documents for a given question"""
        retrieved_docs = state["chroma_db"].similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def _generate(self, state: RAGState) -> dict:
        """Generate answer based on retrieved context"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = f"""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {state["question"]} 
        Context: {docs_content} 
        Answer:
        """
        response = state["llm"].invoke(messages)
        return {"answer": response.content}
    
    def setup_graph(self) -> None:
        """Setup the LangGraph for RAG pipeline"""
        graph_builder = StateGraph(RAGState).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()
    
    def initialize(self, directory_path: str = None, limit: int = 10, 
                  progress_callback: Optional[callable] = None) -> dict:
        """Initialize the complete RAG system"""
        try:
            # Load documents
            self.load_documents(directory_path, limit)
            
            # Chunk documents
            self.chunk_documents()
            
            # Setup vector store with advanced embedding if enabled
            self.setup_vector_store(progress_callback)
            
            # Setup graph
            self.setup_graph()
            
            result = {
                "status": "success",
                "message": "RAG system initialized successfully",
                "documents_info": self.get_documents_info()
            }
            
            # Ajouter les stats d'embedding si disponibles
            if self.embedding_stats:
                result["embedding_stats"] = self.embedding_stats
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to initialize RAG system: {str(e)}",
                "documents_info": {}
            }
    
    def ask_question(self, question: str) -> str:
        """Ask a question to the RAG system"""
        if not self.graph or not self.vector_store:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        response = self.graph.invoke({
            "question": question,
            "chroma_db": self.vector_store,
            "llm": self.llm
        })
        
        return response["answer"]
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready to answer questions"""
        return (self.graph is not None and 
                self.vector_store is not None and 
                len(self.documents) > 0)
    
    def get_embedding_stats(self) -> dict:
        """Get embedding statistics if advanced embedding was used"""
        if self.use_advanced_embedding and self.embedding_manager:
            stats = self.embedding_manager.get_stats()
            stats.update(self.embedding_stats)
            return stats
        return {"message": "Advanced embedding not used"}
    
    def clear_embedding_cache(self):
        """Clear the embedding cache if using advanced embedding"""
        if self.use_advanced_embedding and self.embedding_manager:
            self.embedding_manager.clear_cache()
            return {"message": "Cache cleared"}
        return {"message": "Advanced embedding not used or no cache available"}
        # try:
        #     # Load documents
        #     self.load_documents(directory_path)

        #     # Chunk documents
        #     self.chunk_documents()

        #     # Setup vector store
        #     self.setup_vector_store()

        #     # Setup graph
        #     self.setup_graph()

        #     return {
        #         "status": "success",
        #         "message": "RAG system initialized successfully",
        #         "documents_info": self.get_documents_info(),
        #     }
        # except Exception as e:
        #     return {
        #         "status": "error",
        #         "message": f"Failed to initialize RAG system: {str(e)}",
        #         "documents_info": {},
        #     }

    def ask_question(self, question: str) -> str:
        """Ask a question to the RAG system"""
        if not self.graph or not self.vector_store:
            raise ValueError("RAG system not initialized. Call initialize() first.")

        response = self.graph.invoke(
            {
                "question": question,
            }
        )

        return response["answer"]

    def is_ready(self) -> bool:
        """Check if the RAG system is ready to answer questions"""
        return (
            self.graph is not None
            and self.vector_store is not None
            and len(self.documents) > 0
        )
