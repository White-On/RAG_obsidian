from src.rag_system import RAGSystem
from rich.console import Console

console = Console()


def main():
    """Main function using the new RAG system"""
    console.rule("[bold blue]RAG Application with Google Gemini and LangGraph")
    
    # Initialize RAG system
    rag_system = RAGSystem()
    console.log("Initialized RAG System")
    
    # Initialize the complete system
    result = rag_system.initialize(limit=10)
    
    if result["status"] == "success":
        console.log("[bold green]RAG system initialized successfully")
        
        # Get documents info
        docs_info = rag_system.get_documents_info()
        console.log(f"Loaded {docs_info['total_documents']} documents")
        console.log(f"Created {docs_info['total_chunks']} chunks")
        
        # Test with a question
        question = "What is the third step of a PCA?"
        console.log(f"[bold yellow]Question: {question}")
        
        try:
            answer = rag_system.ask_question(question)
            console.rule("[bold green]Response")
            console.log(answer)
        except Exception as e:
            console.log(f"[bold red]Error: {e}")
    else:
        console.log(f"[bold red]Failed to initialize: {result['message']}")

if __name__ == "__main__":
    main()