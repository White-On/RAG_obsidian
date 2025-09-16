import src.config as config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console
from langchain.chat_models import init_chat_model
import langchain.chat_models.base
from dotenv import load_dotenv

load_dotenv()


console = Console()


def load_text_files_from_directory(directory_path: str) -> List[Document]:
    """Load all .txt files from a directory and return as a list of Documents."""
    # TODO: Add support for other file types (e.g., PDFs, Word docs)
    documents = []
    for file_path in Path(directory_path).rglob("*.md"):
        try:
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        except Exception as e:
            console.log(f"Error loading {file_path}: {e}")
    return documents


# Define state for application LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chroma_db: Chroma
    llm: langchain.chat_models.base.BaseChatModel


# Define application steps
def retrieve(state: State):
    retrieved_docs = state["chroma_db"].similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = f"""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {state["question"]} 
    Context: {docs_content} 
    Answer:
    """
    response = state["llm"].invoke(messages)
    return {"answer": response.content}


def main():
    # TODO: Make a custom LLM wrapper for other LLMs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    console.log("Initialized Google Gemini LLM and Embeddings")

    console.rule("[bold blue]RAG Application with Google Gemini and LangGraph")
    # Load and chunk contents of the different documents
    docs = load_text_files_from_directory(config.FILE_VAULT)[:10]  # Limit to first 10 files for demo
    console.log(f"Loaded {len(docs)} documents from {config.FILE_VAULT.name}")

    # TODO : Make a custom text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5_000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    console.log(f"Split into {len(all_splits)} chunks")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )

    # Index chunks
    # TODO : custom usage of the vector store in order
    # to implement custom behavior like embedding element at creation and modification time
    _ = vector_store.add_documents(documents=all_splits)

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke(
        {"question": "What is glycopolymers?"},
        {"chroma_db": vector_store},
        {"llm": llm},
    )
    console.rule("[bold green]Response")
    console.log(response["answer"])


if __name__ == "__main__":
    main()