import chromadb
from chromadb.config import Settings

class ConversationStore:
    def __init__(self, db_url="http://localhost:8000"):
        # Initialize ChromaDB client and set up the collection
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chroma"))
        self.collection = self.client.create_collection(name="conversations")

    def save_conversation(self, user_id, user_message, agent_response, metadata={}):
        # Save conversation to ChromaDB with timestamps
        timestamp = self.get_current_timestamp()
        conversation_id = f"{user_id}-{timestamp}"
        self.collection.add(
            documents=[user_message, agent_response],
            metadatas=[{**metadata, 'timestamp': timestamp, 'conversation_id': conversation_id}]
        )

    def retrieve_conversation(self, user_id, limit=10):
        # Retrieve conversation history with semantic search
        results = self.collection.query(
            query_documents=[user_id],
            n_results=limit
        )
        return results['documents'], results['metadatas']

    @staticmethod
    def get_current_timestamp():
        from datetime import datetime
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
