#!/usr/bin/env python3
"""
Universal Agentic Django Backend Generator
=========================================

NOW SUPPORTS ALL MODELS! 🎉

Supported providers:
- OpenRouter (300+ models including all free ones)  
- OpenAI (GPT-3.5, GPT-4, GPT-4-turbo)
- Anthropic (Claude-3 family)
- Google (Gemini)
- Mistral, Cohere, Meta, Microsoft, and more!

Usage:
    # Use any model you want!
    python universal_backend_builder.py sample_erd.json --model gpt-4
    python universal_backend_builder.py sample_erd.json --model claude-3-sonnet  
    python universal_backend_builder.py sample_erd.json --model gemini-pro
    python universal_backend_builder.py sample_erd.json --model qwen/qwen3-coder:free
    
    # Or use presets
    python universal_backend_builder.py sample_erd.json --preset free
    python universal_backend_builder.py sample_erd.json --preset premium
    python universal_backend_builder.py sample_erd.json --preset balanced
"""

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import our universal model configuration
try:
    from model_config import (
        ModelManager, 
        auto_configure_models, 
        create_universal_client,
        ModelConfig
    )
except ImportError:
    print("❌ model_config.py not found. Make sure it's in the same directory.")
    sys.exit(1)

class UniversalAgent:
    """Universal agent that works with any AI model/provider."""
    
    def __init__(self, model_name: str, model_manager: Optional[ModelManager] = None):
        self.model_name = model_name
        self.model_manager = model_manager or ModelManager()
        self.model_config = self.model_manager.get_model_config(model_name)
        
        if not self.model_config:
            available = self.model_manager.list_available_models()
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
        
        # Create appropriate client for this model
        self.client = create_universal_client(model_name, self.model_manager)
        print(f"🤖 Initialized {self.model_config.provider} client for {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using the configured model."""
        try:
            # Use model-specific parameters
            generation_params = {
                "model": self.model_config.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.model_config.max_tokens),
                "temperature": kwargs.get("temperature", self.model_config.temperature)
            }
            
            # Handle different providers
            if self.model_config.provider in ["openrouter", "openai"]:
                completion = self.client.chat.completions.create(**generation_params)
                return completion.choices[0].message.content
                
            elif self.model_config.provider == "anthropic":
                # Anthropic has different API structure
                response = self.client.messages.create(
                    model=self.model_config.name,
                    max_tokens=generation_params["max_tokens"],
                    temperature=generation_params["temperature"],
                    messages=generation_params["messages"]
                )
                return response.content[0].text
                
            elif self.model_config.provider == "google":
                # Google has different API structure
                model = self.client.GenerativeModel(self.model_config.name)
                response = model.generate_content(prompt)
                return response.text
                
            else:
                raise ValueError(f"Provider {self.model_config.provider} not implemented yet")
                
        except Exception as e:
            print(f"⚠️  Generation failed with {self.model_name}: {e}")
            raise

class UniversalBackendBuilder:
    """Universal Django backend builder supporting ALL AI models."""
    
    def __init__(self, model_config: Dict[str, str]):
        """
        Initialize with model configuration.
        
        Args:
            model_config: Dict with 'primary', 'fallback', 'alternatives' keys
        """
        self.model_config = model_config
        self.model_manager = ModelManager()
        
        # Initialize agents for different models
        self.primary_agent = UniversalAgent(model_config["primary"], self.model_manager)
        self.fallback_agent = UniversalAgent(model_config["fallback"], self.model_manager) if model_config["fallback"] != model_config["primary"] else self.primary_agent
        
        print(f"🚀 Universal Backend Builder initialized!")
        print(f"   Primary: {model_config['primary']}")
        print(f"   Fallback: {model_config['fallback']}")
        print(f"   Alternatives: {len(model_config.get('alternatives', []))} models")
    
    async def generate_models(self, erd: Dict[str, Any]) -> str:
        """Generate Django models.py file."""
        prompt = f"""
Generate a complete Django models.py file from this ERD.

ERD: {json.dumps(erd, indent=2)}

Requirements:
- Include all necessary imports
- Define models with proper field types
- Add relationships (ForeignKey, ManyToMany)
- Include __str__ methods
- Add Meta classes where appropriate
- Follow Django best practices

Return only the Python code for models.py:
"""
        return await self._generate_with_retry("models.py", prompt)
    
    async def generate_serializers(self, erd: Dict[str, Any]) -> str:
        """Generate Django REST Framework serializers.py file."""
        prompt = f"""
Generate a complete Django REST Framework serializers.py file from this ERD.

ERD: {json.dumps(erd, indent=2)}

Requirements:
- Import all necessary modules
- Create ModelSerializer for each model
- Include all fields in Meta class
- Add validation methods where needed
- Follow DRF best practices

Return only the Python code for serializers.py:
"""
        return await self._generate_with_retry("serializers.py", prompt)
    
    async def generate_views(self, erd: Dict[str, Any]) -> str:
        """Generate Django REST Framework views.py file."""
        prompt = f"""
Generate a complete Django REST Framework views.py file from this ERD.

ERD: {json.dumps(erd, indent=2)}

Requirements:
- Import all necessary modules
- Create ModelViewSet for each model
- Include proper authentication and permissions
- Add filtering and pagination
- Follow DRF best practices

Return only the Python code for views.py:
"""
        return await self._generate_with_retry("views.py", prompt)
    
    async def generate_urls(self, erd: Dict[str, Any]) -> str:
        """Generate Django urls.py file."""
        prompt = f"""
Generate a complete Django urls.py file from this ERD.

ERD: {json.dumps(erd, indent=2)}

Requirements:
- Import all necessary modules
- Use DRF DefaultRouter
- Register all ViewSets
- Include API versioning
- Add documentation URLs

Return only the Python code for urls.py:
"""
        return await self._generate_with_retry("urls.py", prompt)
    
    async def generate_settings(self, erd: Dict[str, Any]) -> str:
        """Generate Django settings.py file."""
        prompt = f"""
Generate a complete Django settings.py file for this project.

ERD: {json.dumps(erd, indent=2)}

Requirements:
- Include all necessary Django apps
- Add Django REST Framework configuration
- Configure JWT authentication
- Set up CORS headers
- Include database configuration
- Add environment variable support

Return only the Python code for settings.py:
"""
        return await self._generate_with_retry("settings.py", prompt)
    
    async def _generate_with_retry(self, file_type: str, prompt: str, max_retries: int = 3) -> str:
        """Generate content with automatic retry and fallback."""
        current_agent = self.primary_agent
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 [{file_type}] Generating with {current_agent.model_name}... (attempt {attempt + 1})")
                result = await current_agent.generate(prompt)
                print(f"✅ [{file_type}] Generated successfully with {current_agent.model_name}")
                return result
                
            except Exception as e:
                print(f"⚠️  [{file_type}] Failed with {current_agent.model_name}: {e}")
                
                if attempt < max_retries - 1:
                    # Try fallback model
                    if current_agent == self.primary_agent and self.fallback_agent != self.primary_agent:
                        current_agent = self.fallback_agent
                        print(f"🔄 [{file_type}] Switching to fallback: {current_agent.model_name}")
                    else:
                        # Try alternative models
                        alternatives = self.model_config.get("alternatives", [])
                        if len(alternatives) > attempt + 1:
                            alt_model = alternatives[attempt + 1]
                            current_agent = UniversalAgent(alt_model, self.model_manager)
                            print(f"🔄 [{file_type}] Switching to alternative: {alt_model}")
                        else:
                            print(f"❌ [{file_type}] No more alternatives available")
                            break
                else:
                    print(f"❌ [{file_type}] All retry attempts failed")
                    raise
        
        # If we get here, all attempts failed
        return f"# ERROR: Failed to generate {file_type}\n# All models failed after {max_retries} attempts"
    
    async def generate_complete_backend(self, erd: Dict[str, Any], output_dir: str = "backend") -> Dict[str, str]:
        """Generate complete Django backend with all files."""
        print(f"🚀 Starting complete backend generation...")
        print(f"📁 Output directory: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate all files in parallel for maximum speed
        print(f"⚡ Generating 5 files in parallel...")
        tasks = [
            self.generate_models(erd),
            self.generate_serializers(erd),
            self.generate_views(erd),
            self.generate_urls(erd),
            self.generate_settings(erd)
        ]
        
        try:
            results = await asyncio.gather(*tasks)
            models_code, serializers_code, views_code, urls_code, settings_code = results
            
            # File mapping
            files = {
                "models.py": models_code,
                "serializers.py": serializers_code,
                "views.py": views_code,
                "urls.py": urls_code,
                "settings.py": settings_code
            }
            
            # Write all files
            for filename, content in files.items():
                file_path = Path(output_dir) / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"💾 Saved: {file_path}")
            
            # Generate additional files
            additional_files = self._generate_additional_files(erd)
            for filename, content in additional_files.items():
                file_path = Path(output_dir) / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"💾 Saved: {file_path}")
            
            files.update(additional_files)
            
            print(f"🎉 Backend generation complete!")
            print(f"📁 {len(files)} files created in {output_dir}/")
            
            return files
            
        except Exception as e:
            print(f"❌ Backend generation failed: {e}")
            raise
    
    def _generate_additional_files(self, erd: Dict[str, Any]) -> Dict[str, str]:
        """Generate additional files like requirements.txt, Dockerfile, etc."""
        return {
            "requirements.txt": """Django>=4.2.0
djangorestframework>=3.14.0
django-cors-headers>=4.0.0
djangorestframework-simplejwt>=5.2.0
python-decouple>=3.8
psycopg2-binary>=2.9.0
gunicorn>=21.2.0
""",
            
            "Dockerfile": """FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "backend.wsgi:application"]
""",
            
            "docker-compose.yml": """version: '3.8'

services:
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      POSTGRES_DB: django_db
      POSTGRES_USER: django_user
      POSTGRES_PASSWORD: django_password
    ports:
      - "5432:5432"

  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    environment:
      - DEBUG=1
      - DATABASE_URL=postgres://django_user:django_password@db:5432/django_db
    depends_on:
      - db

volumes:
  postgres_data:
""",
            
            ".env.example": """# Django settings
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgres://user:password@localhost:5432/dbname

# API Keys (choose what you need)
OPENROUTER_API_KEY=your-openrouter-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
"""
        }

def main():
    """Main entry point with support for ALL models."""
    parser = argparse.ArgumentParser(description="Universal Agentic Django Backend Generator")
    parser.add_argument("erd_file", nargs='?', help="Path to ERD JSON file")
    parser.add_argument("--model", help="Specific model to use (e.g., gpt-4, claude-3-sonnet, qwen/qwen3-coder:free)")
    parser.add_argument("--preset", choices=["free", "premium", "balanced", "coding"], 
                       default="free", help="Model preset to use")
    parser.add_argument("--output", default="backend", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        manager = ModelManager()
        print("🤖 Available Models:")
        print("=" * 50)
        
        print("\n🆓 Free Models:")
        for model in manager.get_free_models():
            config = manager.get_model_config(model)
            print(f"  • {model} ({config.provider})")
        
        print("\n💎 Premium Models:")
        for model in ["gpt-4-turbo", "claude-3-sonnet", "gpt-4", "claude-3-haiku"]:
            if manager.get_model_config(model):
                config = manager.get_model_config(model)
                print(f"  • {model} ({config.provider}) - ${config.cost_per_1k_tokens}/1k tokens")
        
        print("\n🧠 All Coding Models:")
        for model in manager.get_models_by_capability("coding"):
            config = manager.get_model_config(model)
            cost = "Free" if config.cost_per_1k_tokens == 0 else f"${config.cost_per_1k_tokens}/1k"
            print(f"  • {model} ({config.provider}) - {cost}")
        
        print(f"\n💡 Usage examples:")
        print(f"  python {sys.argv[0]} erd.json --model gpt-4")
        print(f"  python {sys.argv[0]} erd.json --model claude-3-sonnet")
        print(f"  python {sys.argv[0]} erd.json --model qwen/qwen3-coder:free")
        print(f"  python {sys.argv[0]} erd.json --preset premium")
        return
    
    # Load ERD file
    if not args.erd_file:
        print(f"❌ ERD file required when not using --list-models")
        sys.exit(1)
    
    if not os.path.exists(args.erd_file):
        print(f"❌ ERD file not found: {args.erd_file}")
        sys.exit(1)
    
    with open(args.erd_file, 'r') as f:
        erd_data = json.load(f)
    
    # Configure models
    if args.model:
        # Use specific model
        model_config = auto_configure_models(args.model)
    else:
        # Use preset
        model_config = auto_configure_models(args.preset)
    
    print(f"🎯 Model Configuration:")
    print(f"   Primary: {model_config['primary']}")
    print(f"   Fallback: {model_config['fallback']}")
    print(f"   Alternatives: {len(model_config.get('alternatives', []))} models")
    
    # Check API keys
    manager = ModelManager()
    primary_config = manager.get_model_config(model_config['primary'])
    api_key = os.getenv(primary_config.api_key_env)
    if not api_key:
        print(f"❌ API key not found. Set environment variable: {primary_config.api_key_env}")
        print(f"💡 Example: export {primary_config.api_key_env}='your-key-here'")
        sys.exit(1)
    
    # Generate backend
    async def run_generation():
        builder = UniversalBackendBuilder(model_config)
        await builder.generate_complete_backend(erd_data, args.output)
    
    print(f"🚀 Starting universal backend generation...")
    asyncio.run(run_generation())
    print(f"🎉 Complete! Check the {args.output}/ directory")

if __name__ == "__main__":
    main() 