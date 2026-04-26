# src/blog_agent_system/persistence/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_server_host="chromadb",
            chroma_server_http_port=8000,
            anonymized_telemetry=False
        ))
        
        # Collections for different knowledge domains
        self.collections = {
            "blog_knowledge": self.client.get_or_create_collection("blog_knowledge"),
            "style_guides": self.client.get_or_create_collection("style_guides"),
            "previous_blogs": self.client.get_or_create_collection("previous_blogs")
        }
    
    def get_collection(self, name: str):
        return self.collections.get(name)