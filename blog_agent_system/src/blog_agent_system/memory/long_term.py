# src/blog_agent_system/memory/long_term.py
import chromadb
from langchain_openai import OpenAIEmbeddings

class LongTermMemory:
    def __init__(self, collection_name: str = "blog_knowledge"):
        self.client = chromadb.HttpClient(host="chromadb", port=8000)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    async def store(self, documents: list[str], metadata: list[dict], ids: list[str]):
        embeddings = await self.embeddings.aembed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    async def retrieve(self, query: str, top_k: int = 5, filter: dict = None) -> list[dict]:
        query_embedding = await self.embeddings.aembed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]