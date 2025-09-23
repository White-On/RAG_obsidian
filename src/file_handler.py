from pathlib import Path


def load_markdown_file(file_path: Path) -> str:
    """Load and return the content of a markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_text_file(file_path: Path) -> str:
    """Load and return the content of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_supported_file(file_path: Path) -> str:
    """Load and return the content of a supported file type."""
    if file_path.suffix == ".md":
        return load_markdown_file(file_path)
    elif file_path.suffix == ".txt":
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")


def load_files_from_directory(directory_path: Path) -> dict:
    """Load and return contents of all supported files in a directory."""
    documents = []
    unsupported_files: list[Path] = []
    file_paths = list(directory_path.rglob("*"))

    for file_path in file_paths:
        if file_path.is_file() and file_path.exists():
            try:
                content = load_supported_file(file_path)
                if len(content.strip()) == 0:
                    continue  # Skip empty files
                documents.append(
                    {
                        "name": file_path.name,
                        "content": content,
                        "source": str(file_path),
                    }
                )
            except ValueError as e:
                unsupported_files.append(file_path)
    print(f"Loaded {len(documents)} documents from {directory_path}")
    if unsupported_files:
        print(f"Skipped {len(unsupported_files)} unsupported files.")
        # for uf in unsupported_files:
        #     print(f" - {uf.name}")

    return documents


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into chunks of specified size with overlap."""
    if chunk_size <= overlap:
        return [text]  # Avoid infinite loop if overlap >= chunk_size

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += chunk_size - overlap

    return chunks
