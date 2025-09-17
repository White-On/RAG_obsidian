# RAG Experiments for obsdian plugin

QUESTIONS:
- When do we embedd the documents ? for now just embedd if not yet in the DB
- for the embedding we stick to an API, same for the LLM
- Obsidian plugin -> React components that trigger python elements ? 
- what about the Obsidian Graph ? gather close element from the graph to or just redirect to it.

## TODO 
- [] better file handle -> metadata of what's already inside Chroma DB
- [] embedding in batch or custom function for embedding 
- [] Embedding on a set of file, do not embedd if elements already in it + Chroma DB
- [] The ask a question to the documents so prompt + LLM chat with streamlit should be enougth for now
- [] See how to integrate this to obsidian embedded in within on the side as chat 
- [] polish project with the questions to solve
