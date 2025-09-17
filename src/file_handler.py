from pathlib import Path

def load_markdown_file(file_path: Path) -> str:
    """Load and return the content of a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_text_file(file_path: Path) -> str:
    """Load and return the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def load_supported_file(file_path: Path) -> str:
    """Load and return the content of a supported file type."""
    if file_path.suffix == '.md':
        return load_markdown_file(file_path)
    elif file_path.suffix == '.txt':
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")

def load_files_from_directory(directory_path: Path) -> dict:
    """Load and return contents of all supported files in a directory."""
    documents = []
    file_paths = list(directory_path.rglob('*'))
    
    for file_path in file_paths:
        if file_path.is_file() and file_path.exists():
            try:
                content = load_supported_file(file_path)
                documents.append({
                    "name": file_path.stem, 
                    "content": content,
                    "source": str(file_path)
                    })
            except ValueError as e:
                print(e)

    return documents

