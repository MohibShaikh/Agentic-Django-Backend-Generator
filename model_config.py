"""
Universal Model Configuration for Agentic Django Backend Generator
================================================================

Supports ALL major AI models and providers:
- OpenRouter (300+ models)
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Meta (Llama)
- Mistral
- Cohere
- And many more!

Usage:
- Set environment variables for API keys
- Choose models in config or command line
- System automatically handles different APIs
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str  # "openrouter", "openai", "anthropic", etc.
    api_key_env: str  # Environment variable name for API key
    base_url: Optional[str] = None  # Custom base URL if needed
    max_tokens: int = 4000
    temperature: float = 0.2
    cost_per_1k_tokens: float = 0.0  # Cost in USD
    context_window: int = 8000
    capabilities: List[str] = None  # ["coding", "reasoning", "multimodal"]
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["coding"]

# ============================================================================
# SUPPORTED MODELS - Add any model you want here!
# ============================================================================

SUPPORTED_MODELS = {
    # ===== FREE MODELS (OpenRouter) =====
    "qwen/qwen3-coder:free": ModelConfig(
        name="qwen/qwen3-coder:free",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=2000,
        cost_per_1k_tokens=0.0,
        context_window=8000,
        capabilities=["coding", "reasoning"]
    ),
    
    "deepseek/deepseek-r1-0528:free": ModelConfig(
        name="deepseek/deepseek-r1-0528:free", 
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=2000,
        cost_per_1k_tokens=0.0,
        context_window=8000,
        capabilities=["coding", "reasoning"]
    ),
    
    "meta-llama/llama-3.2-3b-instruct:free": ModelConfig(
        name="meta-llama/llama-3.2-3b-instruct:free",
        provider="openrouter", 
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=2000,
        cost_per_1k_tokens=0.0,
        context_window=8000,
        capabilities=["coding", "reasoning"]
    ),
    
    "microsoft/phi-3-mini-128k-instruct:free": ModelConfig(
        name="microsoft/phi-3-mini-128k-instruct:free",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY", 
        max_tokens=2000,
        cost_per_1k_tokens=0.0,
        context_window=8000,
        capabilities=["coding"]
    ),
    
    # ===== OPENAI MODELS =====
    "gpt-4": ModelConfig(
        name="gpt-4",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.03,
        context_window=8000,
        capabilities=["coding", "reasoning", "advanced"]
    ),
    
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        provider="openai", 
        api_key_env="OPENAI_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.01,
        context_window=128000,
        capabilities=["coding", "reasoning", "advanced"]
    ),
    
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        provider="openai",
        api_key_env="OPENAI_API_KEY", 
        max_tokens=4000,
        cost_per_1k_tokens=0.001,
        context_window=16000,
        capabilities=["coding", "reasoning"]
    ),
    
    # ===== ANTHROPIC MODELS =====
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet-20240229",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.015,
        context_window=200000,
        capabilities=["coding", "reasoning", "advanced"]
    ),
    
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku-20240307", 
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.00025,
        context_window=200000,
        capabilities=["coding", "reasoning"]
    ),
    
    # ===== GOOGLE MODELS =====
    "gemini-pro": ModelConfig(
        name="gemini-pro",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        max_tokens=2048,
        cost_per_1k_tokens=0.0005,
        context_window=30000,
        capabilities=["coding", "reasoning"]
    ),
    
    # ===== OPENROUTER PREMIUM MODELS =====
    "openai/gpt-4": ModelConfig(
        name="openai/gpt-4",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.03,
        context_window=8000,
        capabilities=["coding", "reasoning", "advanced"]
    ),
    
    "anthropic/claude-3-sonnet": ModelConfig(
        name="anthropic/claude-3-sonnet",
        provider="openrouter", 
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.015,
        context_window=200000,
        capabilities=["coding", "reasoning", "advanced"]
    ),
    
    "google/gemini-pro": ModelConfig(
        name="google/gemini-pro",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=2048,
        cost_per_1k_tokens=0.0005,
        context_window=30000,
        capabilities=["coding", "reasoning"]
    ),
    
    "mistralai/mistral-large": ModelConfig(
        name="mistralai/mistral-large",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY", 
        max_tokens=4000,
        cost_per_1k_tokens=0.008,
        context_window=32000,
        capabilities=["coding", "reasoning"]
    ),
    
    "cohere/command-r": ModelConfig(
        name="cohere/command-r",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.0005,
        context_window=128000,
        capabilities=["coding", "reasoning"]
    ),
}

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

class ModelPresets:
    """Pre-configured model combinations for different use cases."""
    
    # Free models only
    FREE_MODELS = [
        "qwen/qwen3-coder:free",
        "deepseek/deepseek-r1-0528:free", 
        "meta-llama/llama-3.2-3b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free"
    ]
    
    # Best quality (premium)
    PREMIUM_MODELS = [
        "gpt-4-turbo",
        "claude-3-sonnet", 
        "openai/gpt-4",
        "anthropic/claude-3-sonnet"
    ]
    
    # Balanced (good quality, reasonable cost)
    BALANCED_MODELS = [
        "gpt-3.5-turbo",
        "claude-3-haiku",
        "gemini-pro",
        "cohere/command-r"
    ]
    
    # Coding specialists
    CODING_MODELS = [
        "qwen/qwen3-coder:free",
        "deepseek/deepseek-r1-0528:free",
        "gpt-4",
        "claude-3-sonnet"
    ]

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ModelManager:
    """Manage model configurations and provide universal access."""
    
    def __init__(self):
        self.models = SUPPORTED_MODELS
        
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def list_available_models(self, provider: Optional[str] = None, free_only: bool = False) -> List[str]:
        """List all available models, optionally filtered."""
        models = []
        for name, config in self.models.items():
            if provider and config.provider != provider:
                continue
            if free_only and config.cost_per_1k_tokens > 0:
                continue
            models.append(name)
        return sorted(models)
    
    def get_free_models(self) -> List[str]:
        """Get all free models."""
        return self.list_available_models(free_only=True)
    
    def get_models_by_capability(self, capability: str) -> List[str]:
        """Get models that support a specific capability."""
        models = []
        for name, config in self.models.items():
            if capability in config.capabilities:
                models.append(name)
        return sorted(models)
    
    def add_custom_model(self, model_config: ModelConfig) -> None:
        """Add a custom model configuration."""
        self.models[model_config.name] = model_config
    
    def get_recommended_models(self, use_case: str = "balanced") -> List[str]:
        """Get recommended models for different use cases."""
        if use_case == "free":
            return ModelPresets.FREE_MODELS
        elif use_case == "premium":
            return ModelPresets.PREMIUM_MODELS
        elif use_case == "coding":
            return ModelPresets.CODING_MODELS
        else:  # balanced
            return ModelPresets.BALANCED_MODELS

# ============================================================================
# UNIVERSAL CLIENT FACTORY
# ============================================================================

def create_universal_client(model_name: str, model_manager: ModelManager = None) -> Any:
    """Create a client for any supported model/provider."""
    if model_manager is None:
        model_manager = ModelManager()
    
    config = model_manager.get_model_config(model_name)
    if not config:
        raise ValueError(f"Model '{model_name}' not supported. Use ModelManager.list_available_models() to see options.")
    
    # Get API key from environment
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found. Set environment variable: {config.api_key_env}")
    
    # Import and create appropriate client
    if config.provider == "openrouter":
        import openai
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    elif config.provider == "openai":
        import openai
        return openai.OpenAI(api_key=api_key)
    elif config.provider == "anthropic":
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    elif config.provider == "google":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    else:
        raise ValueError(f"Provider '{config.provider}' not supported yet")

# ============================================================================
# EASY CONFIGURATION
# ============================================================================

def auto_configure_models(preference: str = "free") -> Dict[str, str]:
    """Auto-configure models based on user preference."""
    manager = ModelManager()
    
    if preference == "free":
        models = manager.get_free_models()
        if not models:
            raise ValueError("No free models available")
        return {
            "primary": models[0],
            "fallback": models[1] if len(models) > 1 else models[0],
            "alternatives": models
        }
    elif preference == "premium":
        models = ModelPresets.PREMIUM_MODELS
        return {
            "primary": models[0],
            "fallback": models[1] if len(models) > 1 else models[0], 
            "alternatives": models
        }
    elif preference == "balanced":
        models = ModelPresets.BALANCED_MODELS
        return {
            "primary": models[0],
            "fallback": models[1] if len(models) > 1 else models[0],
            "alternatives": models
        }
    else:
        # Custom model name
        if manager.get_model_config(preference):
            return {
                "primary": preference,
                "fallback": ModelPresets.FREE_MODELS[0],
                "alternatives": [preference] + ModelPresets.FREE_MODELS
            }
        else:
            raise ValueError(f"Model '{preference}' not supported")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    
    print("🤖 Available Models:")
    print("==================")
    
    print("\n🆓 Free Models:")
    for model in manager.get_free_models():
        print(f"  • {model}")
    
    print("\n💎 Premium Models:")
    for model in ModelPresets.PREMIUM_MODELS:
        print(f"  • {model}")
    
    print("\n🧠 Coding Models:")
    for model in manager.get_models_by_capability("coding"):
        print(f"  • {model}")
    
    print("\n⚙️ Auto Configuration Examples:")
    print(f"Free setup: {auto_configure_models('free')}")
    print(f"Premium setup: {auto_configure_models('premium')}")
    print(f"Balanced setup: {auto_configure_models('balanced')}")
    
    # Example: Add custom model
    custom_model = ModelConfig(
        name="my-custom-model",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000,
        cost_per_1k_tokens=0.001,
        capabilities=["coding", "custom"]
    )
    manager.add_custom_model(custom_model)
    print(f"\n✅ Added custom model: {custom_model.name}") 