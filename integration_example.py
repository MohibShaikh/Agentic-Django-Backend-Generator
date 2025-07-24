# integration_example.py
"""
Complete example integrating vector DB with feedback system
"""

import asyncio

# Mock imports for demonstration
class MockVectorFeedbackManager:
    def __init__(self, vector_db, embedding_provider):
        self.vector_db = vector_db
        self.embedding_provider = embedding_provider
    
    async def store_feedback(self, **kwargs):
        print(f"Storing feedback: {kwargs['feedback_text']}")
    
    async def find_similar_feedback(self, query, **kwargs):
        # Mock similar feedback
        class MockFeedback:
            def __init__(self, feedback_text, satisfaction_score):
                self.feedback_text = feedback_text
                self.satisfaction_score = satisfaction_score
        
        return [
            MockFeedback("Need async support", 0.3),
            MockFeedback("Add error handling", 0.4)
        ]

class MockVectorProvider:
    def __init__(self, **kwargs):
        pass

class MockEmbeddingProvider:
    def __init__(self, **kwargs):
        pass

def create_vector_provider(provider_type, **kwargs):
    return MockVectorProvider(**kwargs)

def create_embedding_provider(provider_type, **kwargs):
    return MockEmbeddingProvider(**kwargs)

class VectorEnhancedBuilder:
    """Enhanced builder with vector database integration."""
    
    def __init__(self, model_config, vector_config=None):
        self.model_config = model_config
        
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
            
            self.vector_feedback_manager = MockVectorFeedbackManager(
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
    print("Vector Database Integration Example")
    print("=" * 50)
    
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
    
    # Option 2: Weaviate Cloud (commented out)
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
        print("\nVector DB suggestions:")
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
    
    print("\nVector-enhanced feedback system working!")
    
    print("\nTo use with real vector databases:")
    print("1. Install: pip install chromadb sentence-transformers")
    print("2. Replace mock classes with real imports")
    print("3. Set up your chosen vector database")
    print("4. Configure API keys if using cloud services")

if __name__ == "__main__":
    asyncio.run(main())
