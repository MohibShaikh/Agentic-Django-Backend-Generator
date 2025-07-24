#!/usr/bin/env python3
"""
Vector Database Setup Guide for Django Backend Generator
=======================================================

🚀 EASY SETUP GUIDE FOR POPULAR VECTOR DATABASES

This guide shows you how to integrate with:
- Weaviate (most popular, good for beginners)
- Pinecone (scalable cloud solution)
- Qdrant (fast and efficient)
- Chroma (simple local option)

Each with step-by-step setup instructions!
"""

def print_setup_instructions():
    """Print setup instructions for each vector database."""
    
    print("🔗 VECTOR DATABASE SETUP GUIDE")
    print("=" * 50)
    
    print("\n📋 CHOOSE YOUR VECTOR DATABASE:")
    print("1. Weaviate - Great for beginners, excellent docs")
    print("2. Pinecone - Scalable cloud, easy setup")
    print("3. Qdrant - Fast, efficient, good for performance")
    print("4. Chroma - Simple local option, good for testing")
    
    print("\n" + "="*60)
    print("🟢 OPTION 1: WEAVIATE SETUP")
    print("="*60)
    
    print("""
📦 INSTALLATION:
pip install weaviate-client

🌐 CLOUD SETUP (Recommended):
1. Go to https://console.weaviate.cloud/
2. Create free cluster (14-day trial)
3. Get your cluster URL and API key
4. Set environment variables:
   export WEAVIATE_URL="https://your-cluster.weaviate.network"
   export WEAVIATE_API_KEY="your-api-key"

💻 USAGE EXAMPLE:
from vector_db_integration import create_vector_provider, create_embedding_provider

# Create providers
embedding_provider = create_embedding_provider("openai")  # or "huggingface" for free
vector_db = create_vector_provider("weaviate", 
                                 url="https://your-cluster.weaviate.network",
                                 api_key="your-api-key",
                                 embedding_provider=embedding_provider)

# Store feedback
await vector_db.store_feedback(feedback_object)

# Search similar
similar = await vector_db.search_similar("async database operations")
""")
    
    print("\n" + "="*60)
    print("🔵 OPTION 2: PINECONE SETUP")
    print("="*60)
    
    print("""
📦 INSTALLATION:
pip install pinecone-client

🌐 CLOUD SETUP:
1. Go to https://app.pinecone.io/
2. Sign up for free tier (1M vectors free)
3. Create project and get API key
4. Set environment variables:
   export PINECONE_API_KEY="your-api-key"
   export PINECONE_ENVIRONMENT="us-east-1-aws"  # or your region

💻 USAGE EXAMPLE:
# Create providers
embedding_provider = create_embedding_provider("openai")
vector_db = create_vector_provider("pinecone",
                                 api_key="your-api-key",
                                 environment="us-east-1-aws",
                                 embedding_provider=embedding_provider)

# Use same interface as above
await vector_db.store_feedback(feedback_object)
similar = await vector_db.search_similar("error handling views")
""")
    
    print("\n" + "="*60)
    print("🟡 OPTION 3: QDRANT SETUP")
    print("="*60)
    
    print("""
📦 INSTALLATION:
pip install qdrant-client

🌐 CLOUD SETUP:
1. Go to https://cloud.qdrant.io/
2. Create free cluster (1GB free)
3. Get cluster URL and API key
4. Set environment variables:
   export QDRANT_URL="https://your-cluster.qdrant.tech:6333"
   export QDRANT_API_KEY="your-api-key"

🐳 LOCAL SETUP (Docker):
docker run -p 6333:6333 qdrant/qdrant

💻 USAGE EXAMPLE:
# Create providers
embedding_provider = create_embedding_provider("huggingface")  # Free option
vector_db = create_vector_provider("qdrant",
                                 url="https://your-cluster.qdrant.tech:6333",
                                 api_key="your-api-key",
                                 embedding_provider=embedding_provider)

# Store and search
await vector_db.store_feedback(feedback_object)
similar = await vector_db.search_similar("model validation")
""")
    
    print("\n" + "="*60)
    print("🟣 OPTION 4: CHROMA SETUP (EASIEST)")
    print("="*60)
    
    print("""
📦 INSTALLATION:
pip install chromadb

💻 LOCAL SETUP (No cloud needed):
# Automatically creates local database

USAGE EXAMPLE:
# Create providers
embedding_provider = create_embedding_provider("huggingface")  # Free, local
vector_db = create_vector_provider("chroma",
                                 persist_directory="./feedback_db",
                                 embedding_provider=embedding_provider)

# Store and search - same interface
await vector_db.store_feedback(feedback_object)
similar = await vector_db.search_similar("serializer improvements")
""")
    
    print("\n" + "="*60)
    print("🔤 EMBEDDING OPTIONS")
    print("="*60)
    
    print("""
Choose your embedding provider:

1. 🆓 HuggingFace (FREE, Local):
   pip install sentence-transformers
   embedding_provider = create_embedding_provider("huggingface")

2. 💰 OpenAI (PAID, Best Quality):
   pip install openai
   export OPENAI_API_KEY="your-key"
   embedding_provider = create_embedding_provider("openai")

3. 💰 Cohere (PAID, Good Alternative):
   pip install cohere
   export COHERE_API_KEY="your-key"
   embedding_provider = create_embedding_provider("cohere")
""")
    
    print("\n" + "="*60)
    print("⚡ QUICK START EXAMPLE")
    print("="*60)
    
    print("""
# Easiest setup - all local, no API keys needed:

1. Install packages:
   pip install chromadb sentence-transformers

2. Use in your code:
   from vector_db_integration import VectorFeedbackManager, create_vector_provider, create_embedding_provider
   
   # Setup
   embedding_provider = create_embedding_provider("huggingface")
   vector_db = create_vector_provider("chroma", embedding_provider=embedding_provider)
   feedback_manager = VectorFeedbackManager(vector_db, embedding_provider)
   
   # Store feedback
   await feedback_manager.store_feedback(
       file_type="models",
       code_snippet="class User(models.Model): ...",
       feedback_text="Add async support",
       action="reject"
   )
   
   # Find similar
   similar = await feedback_manager.find_similar_feedback("database async")
   for feedback in similar:
       print(f"Similar: {feedback.feedback_text}")

3. That's it! Your feedback system now has semantic search!
""")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("• Beginners: Start with Chroma + HuggingFace (free, local)")
    print("• Production: Use Weaviate or Pinecone cloud")
    print("• Performance: Use Qdrant with optimized embeddings")
    print("• Best Quality: OpenAI embeddings (costs ~$0.0001/1k tokens)")

def create_integration_example():
    """Create a practical integration example."""
    
    example_code = '''
# integration_example.py
"""
Complete example integrating vector DB with feedback system
"""

import asyncio
from vector_db_integration import VectorFeedbackManager, create_vector_provider, create_embedding_provider
from integrated_feedback_system import FeedbackIntegratedBuilder

class VectorEnhancedBuilder(FeedbackIntegratedBuilder):
    """Enhanced builder with vector database integration."""
    
    def __init__(self, model_config, vector_config=None):
        super().__init__(model_config)
        
        if vector_config:
            # Setup vector database
            self.embedding_provider = create_embedding_provider(
                vector_config.get("embedding_provider", "huggingface")
            )
            
            self.vector_db = create_vector_provider(
                vector_config.get("vector_provider", "chroma"),
                embedding_provider=self.embedding_provider,
                **vector_config.get("vector_config", {})
            )
            
            self.vector_feedback_manager = VectorFeedbackManager(
                self.vector_db, self.embedding_provider
            )
        else:
            self.vector_feedback_manager = None
    
    async def get_similar_feedback_suggestions(self, file_type: str, code: str):
        """Get suggestions based on similar feedback."""
        if not self.vector_feedback_manager:
            return []
        
        # Search for similar code patterns
        query = f"{file_type} {code[:200]}"  # First 200 chars as query
        similar_feedback = await self.vector_feedback_manager.find_similar_feedback(
            query, file_type=file_type, limit=3
        )
        
        suggestions = []
        for feedback in similar_feedback:
            if feedback.satisfaction_score < 0.5:  # Low satisfaction = common issue
                suggestions.append(f"Common issue: {feedback.feedback_text}")
        
        return suggestions
    
    async def store_vector_feedback(self, file_type: str, code: str, action: str, 
                                  feedback_text: str = "", satisfaction: float = 0.5):
        """Store feedback in vector database."""
        if self.vector_feedback_manager:
            await self.vector_feedback_manager.store_feedback(
                file_type=file_type,
                code_snippet=code,
                feedback_text=feedback_text,
                action=action,
                satisfaction_score=satisfaction
            )

# Usage example
async def main():
    # Configuration
    model_config = {
        "primary": "gpt-4-turbo",
        "fallback": "gpt-3.5-turbo"
    }
    
    # Vector database config - choose one:
    
    # Option 1: Local (easiest)
    vector_config = {
        "embedding_provider": "huggingface",
        "vector_provider": "chroma",
        "vector_config": {"persist_directory": "./feedback_vectors"}
    }
    
    # Option 2: Weaviate Cloud
    # vector_config = {
    #     "embedding_provider": "openai",
    #     "vector_provider": "weaviate",
    #     "vector_config": {
    #         "url": "https://your-cluster.weaviate.network",
    #         "api_key": "your-api-key"
    #     }
    # }
    
    # Initialize enhanced builder
    builder = VectorEnhancedBuilder(model_config, vector_config)
    
    # Generate with vector-enhanced feedback
    erd = {"User": {"name": "CharField", "email": "EmailField"}}
    
         # Get suggestions before generation
     suggestions = await builder.get_similar_feedback_suggestions("models", "class User")
     if suggestions:
         print("Vector DB suggestions:")
         for suggestion in suggestions:
             print(f"  • {suggestion}")
     
     # Store feedback after generation
     await builder.store_vector_feedback(
         file_type="models",
         code="class User(models.Model): ...",
         action="approve",
         feedback_text="Good model structure",
         satisfaction=0.8
     )
     
     print("✅ Vector-enhanced feedback system working!")
 
 if __name__ == "__main__":
     asyncio.run(main())
 '''
     
     with open("integration_example.py", "w", encoding='utf-8') as f:
         f.write(example_code)
    
    print("💾 Created integration_example.py")
    print("📖 Run with: python integration_example.py")

def show_comparison_table():
    """Show comparison of vector database options."""
    
    print("\n📊 VECTOR DATABASE COMPARISON")
    print("=" * 70)
    
    print("| Feature       | Weaviate | Pinecone | Qdrant | Chroma |")
    print("|---------------|----------|----------|--------|--------|")
    print("| Setup         | Easy     | Easy     | Medium | Easiest|")
    print("| Cost          | Free*    | Free*    | Free*  | Free   |")
    print("| Performance   | High     | High     | Highest| Medium |")
    print("| Scalability   | High     | Highest  | High   | Low    |")
    print("| Local Option  | Yes      | No       | Yes    | Yes    |")
    print("| Cloud Option  | Yes      | Yes      | Yes    | No     |")
    print("| Documentation | Excellent| Good     | Good   | Good   |")
    print("| Best For      | General  | Scale    | Speed  | Local  |")
    
    print("\n* Free tiers available with limitations")
    
    print("\n🎯 QUICK DECISION GUIDE:")
    print("• Just testing? → Chroma")
    print("• Need cloud? → Weaviate or Pinecone")
    print("• Need speed? → Qdrant")
    print("• Need scale? → Pinecone")
    print("• Want simplicity? → Weaviate")

if __name__ == "__main__":
    print_setup_instructions()
    show_comparison_table()
    create_integration_example()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Choose a vector database from the options above")
    print("2. Install the required packages")
    print("3. Set up your API keys (if using cloud)")
    print("4. Run the integration example")
    print("5. Your feedback system now has semantic search!")
    
    print("\n💡 TIP: Start with Chroma + HuggingFace for testing,")
    print("   then upgrade to cloud solution for production.") 