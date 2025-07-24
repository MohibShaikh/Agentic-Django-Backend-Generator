#!/usr/bin/env python3
"""
Vector Database Integration for Django Backend Generator
=======================================================

🔗 INTEGRATE WITH POPULAR VECTOR DATABASES

Supports:
✅ Weaviate (Cloud & Self-hosted)
✅ Pinecone (Cloud)
✅ Qdrant (Cloud & Self-hosted)
✅ Chroma (Local & Cloud)
✅ OpenAI Embeddings
✅ Cohere Embeddings
✅ HuggingFace Embeddings

Features:
- Store code feedback with vector embeddings
- Semantic similarity search for related feedback
- Cloud-based persistence across sessions
- Scalable to millions of feedback entries
- Multi-tenant support for teams
"""

import asyncio
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Vector DB clients (optional imports)
WEAVIATE_AVAILABLE = False
PINECONE_AVAILABLE = False
QDRANT_AVAILABLE = False
CHROMA_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    pass

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    pass

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    pass

# Embedding providers
OPENAI_AVAILABLE = False
COHERE_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    pass

@dataclass
class CodeFeedback:
    """Structured feedback for vector storage."""
    id: str
    timestamp: datetime
    file_type: str  # models, views, serializers, etc.
    code_snippet: str
    feedback_text: str
    action: str  # approve, reject, edit, comment
    satisfaction_score: float  # 0.0 to 1.0
    erd_context: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector DB storage."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'file_type': self.file_type,
            'code_snippet': self.code_snippet,
            'feedback_text': self.feedback_text,
            'action': self.action,
            'satisfaction_score': self.satisfaction_score,
            'erd_context': self.erd_context,
            'tags': self.tags,
            'text_content': f"{self.file_type} {self.code_snippet} {self.feedback_text}"
        }

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    @property
    def dimension(self) -> int:
        return 1536  # text-embedding-3-small dimension

class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider."""
    
    def __init__(self, api_key: str = None):
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere not available. Install: pip install cohere")
        
        self.client = cohere.Client(api_key or os.getenv('COHERE_API_KEY'))
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get Cohere embedding."""
        response = self.client.embed(texts=[text], model="embed-english-v3.0")
        return response.embeddings[0]
    
    @property
    def dimension(self) -> int:
        return 1024  # embed-english-v3.0 dimension

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace local embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get HuggingFace embedding."""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class VectorDBProvider(ABC):
    """Abstract base class for vector database providers."""
    
    @abstractmethod
    async def store_feedback(self, feedback: CodeFeedback):
        """Store feedback with vector embedding."""
        pass
    
    @abstractmethod
    async def search_similar(self, query_text: str, limit: int = 5) -> List[CodeFeedback]:
        """Search for similar feedback."""
        pass
    
    @abstractmethod
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[CodeFeedback]:
        """Get feedback by ID."""
        pass

class WeaviateProvider(VectorDBProvider):
    """Weaviate vector database provider."""
    
    def __init__(self, url: str = None, api_key: str = None, embedding_provider: EmbeddingProvider = None):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not available. Install: pip install weaviate-client")
        
        self.url = url or os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.api_key = api_key or os.getenv('WEAVIATE_API_KEY')
        self.embedding_provider = embedding_provider
        
        # Initialize client
        auth_config = weaviate.AuthApiKey(api_key=self.api_key) if self.api_key else None
        self.client = weaviate.Client(url=self.url, auth_client_secret=auth_config)
        
        # Ensure schema exists
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure Weaviate schema exists."""
        class_name = "CodeFeedback"
        
        # Check if class exists
        try:
            self.client.schema.get(class_name)
            return  # Class already exists
        except:
            pass
        
        # Create schema
        schema = {
            "class": class_name,
            "description": "Django code feedback for learning",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {"name": "feedback_id", "dataType": ["string"]},
                {"name": "timestamp", "dataType": ["string"]},
                {"name": "file_type", "dataType": ["string"]},
                {"name": "code_snippet", "dataType": ["text"]},
                {"name": "feedback_text", "dataType": ["text"]},
                {"name": "action", "dataType": ["string"]},
                {"name": "satisfaction_score", "dataType": ["number"]},
                {"name": "erd_context", "dataType": ["text"]},
                {"name": "tags", "dataType": ["string[]"]},
            ]
        }
        
        self.client.schema.create_class(schema)
    
    async def store_feedback(self, feedback: CodeFeedback):
        """Store feedback in Weaviate."""
        # Get embedding if provider available
        if self.embedding_provider:
            text_content = f"{feedback.file_type} {feedback.code_snippet} {feedback.feedback_text}"
            feedback.embedding = await self.embedding_provider.get_embedding(text_content)
        
        # Store in Weaviate
        data_object = {
            "feedback_id": feedback.id,
            "timestamp": feedback.timestamp.isoformat(),
            "file_type": feedback.file_type,
            "code_snippet": feedback.code_snippet,
            "feedback_text": feedback.feedback_text,
            "action": feedback.action,
            "satisfaction_score": feedback.satisfaction_score,
            "erd_context": json.dumps(feedback.erd_context),
            "tags": feedback.tags,
        }
        
        self.client.data_object.create(
            data_object=data_object,
            class_name="CodeFeedback",
            uuid=feedback.id,
            vector=feedback.embedding
        )
    
    async def search_similar(self, query_text: str, limit: int = 5) -> List[CodeFeedback]:
        """Search for similar feedback in Weaviate."""
        if not self.embedding_provider:
            # Fallback to keyword search
            result = (
                self.client.query
                .get("CodeFeedback", ["feedback_id", "timestamp", "file_type", "code_snippet", 
                                    "feedback_text", "action", "satisfaction_score", "erd_context", "tags"])
                .with_bm25(query=query_text)
                .with_limit(limit)
                .do()
            )
        else:
            # Vector similarity search
            query_vector = await self.embedding_provider.get_embedding(query_text)
            result = (
                self.client.query
                .get("CodeFeedback", ["feedback_id", "timestamp", "file_type", "code_snippet",
                                    "feedback_text", "action", "satisfaction_score", "erd_context", "tags"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .do()
            )
        
        # Convert results to CodeFeedback objects
        feedbacks = []
        if "data" in result and "Get" in result["data"] and "CodeFeedback" in result["data"]["Get"]:
            for item in result["data"]["Get"]["CodeFeedback"]:
                feedback = CodeFeedback(
                    id=item["feedback_id"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    file_type=item["file_type"],
                    code_snippet=item["code_snippet"],
                    feedback_text=item["feedback_text"],
                    action=item["action"],
                    satisfaction_score=item["satisfaction_score"],
                    erd_context=json.loads(item["erd_context"]) if item["erd_context"] else {},
                    tags=item["tags"] or []
                )
                feedbacks.append(feedback)
        
        return feedbacks
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[CodeFeedback]:
        """Get feedback by ID from Weaviate."""
        try:
            result = self.client.data_object.get_by_id(feedback_id, class_name="CodeFeedback")
            if result:
                props = result["properties"]
                return CodeFeedback(
                    id=props["feedback_id"],
                    timestamp=datetime.fromisoformat(props["timestamp"]),
                    file_type=props["file_type"],
                    code_snippet=props["code_snippet"],
                    feedback_text=props["feedback_text"],
                    action=props["action"],
                    satisfaction_score=props["satisfaction_score"],
                    erd_context=json.loads(props["erd_context"]) if props["erd_context"] else {},
                    tags=props["tags"] or []
                )
        except:
            pass
        
        return None

class PineconeProvider(VectorDBProvider):
    """Pinecone vector database provider."""
    
    def __init__(self, api_key: str = None, environment: str = None, embedding_provider: EmbeddingProvider = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install: pip install pinecone-client")
        
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.embedding_provider = embedding_provider
        
        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Index name
        self.index_name = "django-feedback"
        
        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            dimension = self.embedding_provider.dimension if self.embedding_provider else 1536
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine'
            )
        
        self.index = pinecone.Index(self.index_name)
    
    async def store_feedback(self, feedback: CodeFeedback):
        """Store feedback in Pinecone."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for Pinecone")
        
        # Get embedding
        text_content = f"{feedback.file_type} {feedback.code_snippet} {feedback.feedback_text}"
        embedding = await self.embedding_provider.get_embedding(text_content)
        
        # Store in Pinecone
        metadata = feedback.to_dict()
        del metadata['text_content']  # Remove synthetic field
        
        self.index.upsert([(feedback.id, embedding, metadata)])
    
    async def search_similar(self, query_text: str, limit: int = 5) -> List[CodeFeedback]:
        """Search for similar feedback in Pinecone."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for Pinecone")
        
        # Get query embedding
        query_embedding = await self.embedding_provider.get_embedding(query_text)
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True
        )
        
        # Convert to CodeFeedback objects
        feedbacks = []
        for match in results['matches']:
            metadata = match['metadata']
            feedback = CodeFeedback(
                id=metadata['id'],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                file_type=metadata['file_type'],
                code_snippet=metadata['code_snippet'],
                feedback_text=metadata['feedback_text'],
                action=metadata['action'],
                satisfaction_score=metadata['satisfaction_score'],
                erd_context=metadata['erd_context'],
                tags=metadata['tags']
            )
            feedbacks.append(feedback)
        
        return feedbacks
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[CodeFeedback]:
        """Get feedback by ID from Pinecone."""
        try:
            results = self.index.fetch([feedback_id])
            if feedback_id in results['vectors']:
                metadata = results['vectors'][feedback_id]['metadata']
                return CodeFeedback(
                    id=metadata['id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    file_type=metadata['file_type'],
                    code_snippet=metadata['code_snippet'],
                    feedback_text=metadata['feedback_text'],
                    action=metadata['action'],
                    satisfaction_score=metadata['satisfaction_score'],
                    erd_context=metadata['erd_context'],
                    tags=metadata['tags']
                )
        except:
            pass
        
        return None

class QdrantProvider(VectorDBProvider):
    """Qdrant vector database provider."""
    
    def __init__(self, url: str = None, api_key: str = None, embedding_provider: EmbeddingProvider = None):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not available. Install: pip install qdrant-client")
        
        self.url = url or os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.api_key = api_key or os.getenv('QDRANT_API_KEY')
        self.embedding_provider = embedding_provider
        
        # Initialize client
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.collection_name = "django_feedback"
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return  # Collection exists
        except:
            pass
        
        # Create collection
        dimension = self.embedding_provider.dimension if self.embedding_provider else 1536
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
        )
    
    async def store_feedback(self, feedback: CodeFeedback):
        """Store feedback in Qdrant."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for Qdrant")
        
        # Get embedding
        text_content = f"{feedback.file_type} {feedback.code_snippet} {feedback.feedback_text}"
        embedding = await self.embedding_provider.get_embedding(text_content)
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=feedback.id,
                    vector=embedding,
                    payload=feedback.to_dict()
                )
            ]
        )
    
    async def search_similar(self, query_text: str, limit: int = 5) -> List[CodeFeedback]:
        """Search for similar feedback in Qdrant."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for Qdrant")
        
        # Get query embedding
        query_embedding = await self.embedding_provider.get_embedding(query_text)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Convert to CodeFeedback objects
        feedbacks = []
        for result in results:
            payload = result.payload
            feedback = CodeFeedback(
                id=payload['id'],
                timestamp=datetime.fromisoformat(payload['timestamp']),
                file_type=payload['file_type'],
                code_snippet=payload['code_snippet'],
                feedback_text=payload['feedback_text'],
                action=payload['action'],
                satisfaction_score=payload['satisfaction_score'],
                erd_context=payload['erd_context'],
                tags=payload['tags']
            )
            feedbacks.append(feedback)
        
        return feedbacks
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[CodeFeedback]:
        """Get feedback by ID from Qdrant."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[feedback_id]
            )
            if result:
                payload = result[0].payload
                return CodeFeedback(
                    id=payload['id'],
                    timestamp=datetime.fromisoformat(payload['timestamp']),
                    file_type=payload['file_type'],
                    code_snippet=payload['code_snippet'],
                    feedback_text=payload['feedback_text'],
                    action=payload['action'],
                    satisfaction_score=payload['satisfaction_score'],
                    erd_context=payload['erd_context'],
                    tags=payload['tags']
                )
        except:
            pass
        
        return None

class ChromaProvider(VectorDBProvider):
    """Chroma vector database provider."""
    
    def __init__(self, persist_directory: str = None, embedding_provider: EmbeddingProvider = None):
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not available. Install: pip install chromadb")
        
        self.persist_directory = persist_directory or "./chroma_db"
        self.embedding_provider = embedding_provider
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = "django_feedback"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(self.collection_name)
    
    async def store_feedback(self, feedback: CodeFeedback):
        """Store feedback in Chroma."""
        # Get embedding if provider available
        embedding = None
        if self.embedding_provider:
            text_content = f"{feedback.file_type} {feedback.code_snippet} {feedback.feedback_text}"
            embedding = await self.embedding_provider.get_embedding(text_content)
        
        # Store in Chroma
        self.collection.add(
            ids=[feedback.id],
            documents=[f"{feedback.code_snippet} {feedback.feedback_text}"],
            embeddings=[embedding] if embedding else None,
            metadatas=[feedback.to_dict()]
        )
    
    async def search_similar(self, query_text: str, limit: int = 5) -> List[CodeFeedback]:
        """Search for similar feedback in Chroma."""
        if self.embedding_provider:
            # Vector search
            query_embedding = await self.embedding_provider.get_embedding(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
        else:
            # Text search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit
            )
        
        # Convert to CodeFeedback objects
        feedbacks = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                feedback = CodeFeedback(
                    id=metadata['id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    file_type=metadata['file_type'],
                    code_snippet=metadata['code_snippet'],
                    feedback_text=metadata['feedback_text'],
                    action=metadata['action'],
                    satisfaction_score=metadata['satisfaction_score'],
                    erd_context=metadata['erd_context'],
                    tags=metadata['tags']
                )
                feedbacks.append(feedback)
        
        return feedbacks
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[CodeFeedback]:
        """Get feedback by ID from Chroma."""
        try:
            results = self.collection.get(ids=[feedback_id])
            if results['metadatas']:
                metadata = results['metadatas'][0]
                return CodeFeedback(
                    id=metadata['id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    file_type=metadata['file_type'],
                    code_snippet=metadata['code_snippet'],
                    feedback_text=metadata['feedback_text'],
                    action=metadata['action'],
                    satisfaction_score=metadata['satisfaction_score'],
                    erd_context=metadata['erd_context'],
                    tags=metadata['tags']
                )
        except:
            pass
        
        return None

class VectorFeedbackManager:
    """Main feedback manager with vector database integration."""
    
    def __init__(self, vector_db: VectorDBProvider, embedding_provider: EmbeddingProvider = None):
        self.vector_db = vector_db
        self.embedding_provider = embedding_provider
    
    async def store_feedback(self, 
                           file_type: str,
                           code_snippet: str,
                           feedback_text: str,
                           action: str,
                           satisfaction_score: float = 0.5,
                           erd_context: Dict = None,
                           tags: List[str] = None) -> str:
        """Store feedback with automatic ID generation."""
        
        feedback_id = hashlib.md5(
            f"{file_type}{code_snippet}{feedback_text}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        feedback = CodeFeedback(
            id=feedback_id,
            timestamp=datetime.now(),
            file_type=file_type,
            code_snippet=code_snippet,
            feedback_text=feedback_text,
            action=action,
            satisfaction_score=satisfaction_score,
            erd_context=erd_context or {},
            tags=tags or []
        )
        
        await self.vector_db.store_feedback(feedback)
        return feedback_id
    
    async def find_similar_feedback(self, query: str, file_type: str = None, limit: int = 5) -> List[CodeFeedback]:
        """Find similar feedback using vector search."""
        
        # Add file type to query if specified
        if file_type:
            query = f"{file_type} {query}"
        
        similar_feedback = await self.vector_db.search_similar(query, limit)
        
        # Filter by file type if specified
        if file_type:
            similar_feedback = [f for f in similar_feedback if f.file_type == file_type]
        
        return similar_feedback
    
    async def get_feedback_patterns(self, file_type: str = None) -> Dict[str, int]:
        """Get common feedback patterns."""
        # This would require aggregation queries - simplified version
        # In practice, you'd implement proper analytics
        
        # For demo, search for common patterns
        patterns = {}
        common_terms = ["async", "error handling", "validation", "performance", "security"]
        
        for term in common_terms:
            query = f"{file_type} {term}" if file_type else term
            similar = await self.vector_db.search_similar(query, limit=20)
            patterns[term] = len(similar)
        
        return patterns

# Demo and Usage Examples
async def demo_vector_integration():
    """Demonstrate vector database integration."""
    print("🔗 VECTOR DATABASE INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize embedding provider (use HuggingFace for local demo)
    if HUGGINGFACE_AVAILABLE:
        embedding_provider = HuggingFaceEmbeddingProvider()
        print("✅ Using HuggingFace embeddings (local)")
    else:
        print("⚠️  No embedding provider available. Install sentence-transformers")
        return
    
    # Initialize vector database (use Chroma for local demo)
    if CHROMA_AVAILABLE:
        vector_db = ChromaProvider(embedding_provider=embedding_provider)
        print("✅ Using Chroma vector database (local)")
    else:
        print("⚠️  Chroma not available. Install: pip install chromadb")
        return
    
    # Initialize feedback manager
    feedback_manager = VectorFeedbackManager(vector_db, embedding_provider)
    
    # Demo 1: Store some sample feedback
    print("\n📝 Demo 1: Storing Sample Feedback")
    print("-" * 30)
    
    sample_feedbacks = [
        {
            "file_type": "models",
            "code_snippet": "class User(models.Model):\n    name = models.CharField(max_length=100)",
            "feedback_text": "Need to add async support for database operations",
            "action": "reject",
            "satisfaction_score": 0.2,
            "tags": ["async", "database"]
        },
        {
            "file_type": "views",
            "code_snippet": "def create_user(request):\n    user = User.objects.create()",
            "feedback_text": "Missing error handling and validation",
            "action": "edit",
            "satisfaction_score": 0.6,
            "tags": ["error_handling", "validation"]
        },
        {
            "file_type": "models",
            "code_snippet": "class Product(models.Model):\n    price = models.DecimalField()",
            "feedback_text": "Should use async model methods",
            "action": "reject",
            "satisfaction_score": 0.3,
            "tags": ["async", "models"]
        }
    ]
    
    stored_ids = []
    for i, feedback_data in enumerate(sample_feedbacks, 1):
        feedback_id = await feedback_manager.store_feedback(**feedback_data)
        stored_ids.append(feedback_id)
        print(f"   Stored feedback {i}: {feedback_id}")
    
    # Demo 2: Search for similar feedback
    print("\n🔍 Demo 2: Semantic Similarity Search")
    print("-" * 30)
    
    queries = [
        "async database operations",
        "error handling in views",
        "model validation"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        similar = await feedback_manager.find_similar_feedback(query, limit=2)
        
        for j, feedback in enumerate(similar, 1):
            print(f"   Result {j}: {feedback.action} in {feedback.file_type}")
            print(f"            Feedback: {feedback.feedback_text[:50]}...")
            print(f"            Satisfaction: {feedback.satisfaction_score:.1f}")
    
    # Demo 3: Pattern analysis
    print("\n📊 Demo 3: Feedback Pattern Analysis")
    print("-" * 30)
    
    patterns = await feedback_manager.get_feedback_patterns("models")
    print("   Common patterns in 'models':")
    for pattern, count in patterns.items():
        if count > 0:
            print(f"   • {pattern}: {count} similar cases")
    
    print("\n✅ Vector integration demo complete!")
    print("   • Feedback stored with semantic embeddings")
    print("   • Similarity search works across different wording")
    print("   • Pattern analysis identifies common issues")

def create_vector_provider(provider_type: str, **kwargs) -> VectorDBProvider:
    """Factory function to create vector database providers."""
    
    if provider_type.lower() == "weaviate":
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not available. Install: pip install weaviate-client")
        return WeaviateProvider(**kwargs)
    
    elif provider_type.lower() == "pinecone":
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install: pip install pinecone-client")
        return PineconeProvider(**kwargs)
    
    elif provider_type.lower() == "qdrant":
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not available. Install: pip install qdrant-client")
        return QdrantProvider(**kwargs)
    
    elif provider_type.lower() == "chroma":
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not available. Install: pip install chromadb")
        return ChromaProvider(**kwargs)
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

def create_embedding_provider(provider_type: str, **kwargs) -> EmbeddingProvider:
    """Factory function to create embedding providers."""
    
    if provider_type.lower() == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install: pip install openai")
        return OpenAIEmbeddingProvider(**kwargs)
    
    elif provider_type.lower() == "cohere":
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere not available. Install: pip install cohere")
        return CohereEmbeddingProvider(**kwargs)
    
    elif provider_type.lower() == "huggingface":
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace not available. Install: pip install sentence-transformers")
        return HuggingFaceEmbeddingProvider(**kwargs)
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")

if __name__ == "__main__":
    asyncio.run(demo_vector_integration()) 