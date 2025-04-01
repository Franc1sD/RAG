import chromadb
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings


load_dotenv()
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

# Embedding function for Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"
        
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            )
        )
        return [e.values for e in response.embeddings]





if __name__ == "__main__":
    api_key = os.getenv('GOOGLE_API_KEY')
    client = genai.Client(api_key=api_key)

    # Open and read documents
    documents = []
    folder = "Documents"
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(content.strip())
    
    # Encode Documents
    DB_NAME = "my_chroma_db"
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True 

    chroma_client = chromadb.Client()
    db = chroma_client.get_or_create_collection(
        name=DB_NAME,
        embedding_function=embed_fn,
        metadata={"description": "My Chroma DB for RAG"}
    )
    db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    # Encode Query
    embed_fn.document_mode = False
    query = "How do you use the touchscreen to play music"
    result = db.query(
        query_texts=[query],
        n_results=3
    )
    [all_passages] = result["documents"]
