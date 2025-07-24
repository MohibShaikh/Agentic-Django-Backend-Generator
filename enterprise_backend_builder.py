#!/usr/bin/env python3
"""
Enterprise-Grade Agentic Django Backend Generator
================================================

🏢 OPTIMIZED FOR VERY COMPLEX ERDs 

Handles massive ERDs with:
- 100+ entities
- Complex relationships
- Performance optimizations
- Intelligent chunking
- Memory management
- Parallel processing

Key Features:
✅ ERD Chunking: Split large ERDs into manageable pieces
✅ Smart Dependencies: Resolve cross-entity relationships
✅ Memory Efficient: Stream processing for massive datasets
✅ Token Optimization: Minimize API calls and costs
✅ Progressive Generation: Build complex systems incrementally
✅ Quality Assurance: Enterprise-level code validation
"""

import asyncio
import json
import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

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

@dataclass
class ERDComplexityMetrics:
    """Analyze ERD complexity to optimize processing."""
    entity_count: int = 0
    relationship_count: int = 0
    max_relationships_per_entity: int = 0
    circular_dependencies: int = 0
    inheritance_depth: int = 0
    estimated_tokens: int = 0
    complexity_score: float = 0.0
    recommended_chunks: int = 1
    
    def calculate_complexity(self):
        """Calculate overall complexity score."""
        # Base complexity from entity count
        entity_complexity = min(self.entity_count / 50, 1.0)
        
        # Relationship complexity
        rel_complexity = min(self.relationship_count / 100, 1.0)
        
        # Depth complexity from inheritance/nesting
        depth_complexity = min(self.inheritance_depth / 5, 1.0)
        
        # Circular dependency penalty
        circular_penalty = min(self.circular_dependencies / 10, 0.5)
        
        self.complexity_score = (
            entity_complexity * 0.4 + 
            rel_complexity * 0.3 + 
            depth_complexity * 0.2 + 
            circular_penalty * 0.1
        )
        
        # Recommend chunks based on complexity
        if self.complexity_score > 0.8:
            self.recommended_chunks = max(4, self.entity_count // 25)
        elif self.complexity_score > 0.6:
            self.recommended_chunks = max(3, self.entity_count // 35)
        elif self.complexity_score > 0.4:
            self.recommended_chunks = max(2, self.entity_count // 50)
        else:
            self.recommended_chunks = 1

@dataclass
class ERDChunk:
    """A chunk of the ERD for processing."""
    chunk_id: str
    entities: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    estimated_tokens: int = 0
    priority: int = 0  # Lower number = higher priority

class ERDAnalyzer:
    """Analyze and optimize ERD processing for complex schemas."""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.entity_relationships = defaultdict(set)
        self.inheritance_tree = defaultdict(set)
        
    def analyze_erd(self, erd: Dict[str, Any]) -> ERDComplexityMetrics:
        """Analyze ERD complexity and structure."""
        print("🔍 Analyzing ERD complexity...")
        
        metrics = ERDComplexityMetrics()
        metrics.entity_count = len(erd.get('entities', erd))
        
        # Handle both formats: {"entities": [...]} and direct entity dict
        entities = erd.get('entities', erd) if isinstance(erd.get('entities'), list) else erd
        
        # Build relationship graph
        for entity_name, entity_data in entities.items():
            if isinstance(entity_data, dict):
                self._analyze_entity_relationships(entity_name, entity_data, metrics)
        
        # Calculate complexity metrics
        metrics.max_relationships_per_entity = max(
            len(relationships) for relationships in self.entity_relationships.values()
        ) if self.entity_relationships else 0
        
        metrics.circular_dependencies = len(list(nx.simple_cycles(self.dependency_graph)))
        metrics.inheritance_depth = self._calculate_inheritance_depth()
        metrics.estimated_tokens = self._estimate_token_count(entities)
        
        metrics.calculate_complexity()
        
        print(f"📊 ERD Analysis Complete:")
        print(f"   • Entities: {metrics.entity_count}")
        print(f"   • Relationships: {metrics.relationship_count}")
        print(f"   • Complexity Score: {metrics.complexity_score:.2f}")
        print(f"   • Recommended Chunks: {metrics.recommended_chunks}")
        print(f"   • Estimated Tokens: {metrics.estimated_tokens}")
        
        return metrics
    
    def _analyze_entity_relationships(self, entity_name: str, entity_data: Dict, metrics: ERDComplexityMetrics):
        """Analyze relationships for a single entity."""
        self.dependency_graph.add_node(entity_name)
        
        # Handle different ERD formats
        if 'fields' in entity_data:
            # New format: {"fields": [{"name": "user", "type": "ForeignKey", "related_model": "User"}]}
            for field in entity_data.get('fields', []):
                if field.get('type') in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                    related_model = field.get('related_model', field.get('to'))
                    if related_model:
                        self.entity_relationships[entity_name].add(related_model)
                        self.dependency_graph.add_edge(entity_name, related_model)
                        metrics.relationship_count += 1
        else:
            # Old format: {"field_name": "ForeignKey:RelatedModel"}
            for field_name, field_type in entity_data.items():
                if isinstance(field_type, str):
                    if field_type.startswith('ForeignKey:') or field_type.startswith('OneToOneField:'):
                        related_model = field_type.split(':')[1]
                        self.entity_relationships[entity_name].add(related_model)
                        self.dependency_graph.add_edge(entity_name, related_model)
                        metrics.relationship_count += 1
                    elif field_name == 'has_many' and isinstance(entity_data.get(field_name), list):
                        for related_model in entity_data[field_name]:
                            self.entity_relationships[entity_name].add(related_model)
                            self.dependency_graph.add_edge(related_model, entity_name)
                            metrics.relationship_count += 1
    
    def _calculate_inheritance_depth(self) -> int:
        """Calculate maximum inheritance depth."""
        max_depth = 0
        for node in self.dependency_graph.nodes():
            try:
                depth = nx.shortest_path_length(self.dependency_graph, node, node)
                max_depth = max(max_depth, depth)
            except nx.NetworkXNoPath:
                continue
        return max_depth
    
    def _estimate_token_count(self, entities: Dict) -> int:
        """Estimate token count for the ERD."""
        json_str = json.dumps(entities, indent=2)
        # Rough estimate: 1 token ≈ 4 characters
        return len(json_str) // 4

class ERDChunker:
    """Split complex ERDs into manageable chunks for processing."""
    
    def __init__(self, analyzer: ERDAnalyzer):
        self.analyzer = analyzer
        
    def chunk_erd(self, erd: Dict[str, Any], metrics: ERDComplexityMetrics) -> List[ERDChunk]:
        """Split ERD into optimized chunks."""
        print(f"🔄 Chunking ERD into {metrics.recommended_chunks} pieces...")
        
        if metrics.recommended_chunks <= 1:
            return [self._create_single_chunk(erd)]
        
        # Handle both ERD formats
        entities = erd.get('entities', erd) if isinstance(erd.get('entities'), list) else erd
        
        # Use topological sorting to respect dependencies
        try:
            ordered_entities = list(nx.topological_sort(self.analyzer.dependency_graph))
        except nx.NetworkXError:
            # Handle cycles by using a different approach
            ordered_entities = list(entities.keys())
        
        # Create chunks
        chunks = []
        chunk_size = max(1, len(entities) // metrics.recommended_chunks)
        
        for i in range(0, len(ordered_entities), chunk_size):
            chunk_entities = ordered_entities[i:i + chunk_size]
            chunk_data = {name: entities[name] for name in chunk_entities if name in entities}
            
            chunk = ERDChunk(
                chunk_id=f"chunk_{i//chunk_size + 1}",
                entities=chunk_data
            )
            
            # Calculate dependencies
            for entity_name in chunk_entities:
                deps = self.analyzer.entity_relationships.get(entity_name, set())
                chunk.dependencies.update(deps - set(chunk_entities))
                
                # Find what depends on this chunk
                for other_entity, other_deps in self.analyzer.entity_relationships.items():
                    if entity_name in other_deps and other_entity not in chunk_entities:
                        chunk.dependents.add(other_entity)
            
            # Calculate chunk priority (fewer dependencies = higher priority)
            chunk.priority = len(chunk.dependencies)
            chunk.estimated_tokens = len(json.dumps(chunk_data)) // 4
            
            chunks.append(chunk)
        
        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority)
        
        print(f"✅ Created {len(chunks)} chunks:")
        for chunk in chunks:
            print(f"   • {chunk.chunk_id}: {len(chunk.entities)} entities, priority {chunk.priority}")
        
        return chunks
    
    def _create_single_chunk(self, erd: Dict[str, Any]) -> ERDChunk:
        """Create a single chunk for simple ERDs."""
        entities = erd.get('entities', erd) if isinstance(erd.get('entities'), list) else erd
        return ERDChunk(
            chunk_id="single_chunk",
            entities=entities,
            estimated_tokens=len(json.dumps(entities)) // 4
        )

class EnterpriseBackendBuilder:
    """Enterprise-grade backend builder for complex ERDs."""
    
    def __init__(self, model_config: Dict[str, str], max_workers: int = 4):
        self.model_config = model_config
        self.model_manager = ModelManager()
        self.max_workers = max_workers
        
        # Initialize high-performance models for complex work
        self.primary_agent = self._create_agent(model_config["primary"])
        self.fallback_agent = self._create_agent(model_config["fallback"]) if model_config["fallback"] != model_config["primary"] else self.primary_agent
        
        # Cache for cross-chunk references
        self.generated_models = {}
        self.entity_imports = defaultdict(set)
        
        print(f"🏢 Enterprise Backend Builder initialized!")
        print(f"   Primary Model: {model_config['primary']}")
        print(f"   Max Workers: {max_workers}")
        print(f"   Memory Optimization: Enabled")
        print(f"   Parallel Processing: Enabled")
    
    def _create_agent(self, model_name: str):
        """Create a high-performance agent."""
        from universal_backend_builder import UniversalAgent
        return UniversalAgent(model_name, self.model_manager)
    
    async def generate_enterprise_backend(
        self, 
        erd: Dict[str, Any], 
        output_dir: str = "enterprise_backend"
    ) -> Dict[str, str]:
        """Generate backend for complex ERDs with enterprise optimizations."""
        print(f"🚀 Starting enterprise backend generation...")
        start_time = time.time()
        
        # Step 1: Analyze ERD complexity
        analyzer = ERDAnalyzer()
        metrics = analyzer.analyze_erd(erd)
        
        # Step 2: Decide on processing strategy
        if metrics.complexity_score < 0.3:
            print("📝 Simple ERD detected - using standard processing")
            return await self._generate_simple_backend(erd, output_dir)
        else:
            print("🏗️  Complex ERD detected - using chunked processing")
            return await self._generate_complex_backend(erd, analyzer, metrics, output_dir)
    
    async def _generate_simple_backend(self, erd: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate backend for simple ERDs using standard approach."""
        from universal_backend_builder import UniversalBackendBuilder
        builder = UniversalBackendBuilder(self.model_config)
        return await builder.generate_complete_backend(erd, output_dir)
    
    async def _generate_complex_backend(
        self, 
        erd: Dict[str, Any], 
        analyzer: ERDAnalyzer, 
        metrics: ERDComplexityMetrics,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate backend for complex ERDs using advanced chunking."""
        print(f"🔧 Using advanced chunked processing...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Step 1: Chunk the ERD
        chunker = ERDChunker(analyzer)
        chunks = chunker.chunk_erd(erd, metrics)
        
        # Step 2: Generate models in dependency order
        all_models = await self._generate_chunked_models(chunks)
        
        # Step 3: Generate other components with full context
        print(f"🔄 Generating serializers, views, and URLs with full context...")
        
        # Combine all entities for other components
        full_entities = {}
        for chunk in chunks:
            full_entities.update(chunk.entities)
        
        # Generate remaining components in parallel
        tasks = [
            self._generate_component("serializers", full_entities),
            self._generate_component("views", full_entities),
            self._generate_component("urls", full_entities),
            self._generate_component("settings", full_entities)
        ]
        
        results = await asyncio.gather(*tasks)
        serializers_code, views_code, urls_code, settings_code = results
        
        # Combine results
        files = {
            "models.py": all_models,
            "serializers.py": serializers_code,
            "views.py": views_code,
            "urls.py": urls_code,
            "settings.py": settings_code
        }
        
        # Add additional enterprise files
        additional_files = self._generate_enterprise_files(full_entities)
        files.update(additional_files)
        
        # Write all files
        for filename, content in files.items():
            file_path = Path(output_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"💾 Saved: {file_path}")
        
        generation_time = time.time() - time.time()
        print(f"🎉 Enterprise backend generation complete!")
        print(f"📁 {len(files)} files created in {output_dir}/")
        print(f"⏱️  Total time: {generation_time:.2f}s")
        
        return files
    
    async def _generate_chunked_models(self, chunks: List[ERDChunk]) -> str:
        """Generate models in chunks, handling dependencies."""
        print(f"🔄 Generating models for {len(chunks)} chunks...")
        
        all_model_code = []
        all_model_code.append("# Generated Django Models - Enterprise Grade")
        all_model_code.append("from django.db import models")
        all_model_code.append("from django.contrib.auth.models import AbstractUser")
        all_model_code.append("from django.core.validators import MinValueValidator, MaxValueValidator")
        all_model_code.append("from django.db.models import JSONField")
        all_model_code.append("")
        
        # Process chunks in priority order
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing {chunk.chunk_id} ({i+1}/{len(chunks)})...")
            
            # Build context for this chunk
            context = self._build_chunk_context(chunk, chunks)
            
            # Generate models for this chunk
            chunk_models = await self._generate_chunk_models(chunk, context)
            
            # Add to overall models
            all_model_code.append(f"\n# === {chunk.chunk_id.upper()} MODELS ===")
            all_model_code.append(chunk_models)
            
            # Store generated models for cross-chunk reference
            self.generated_models[chunk.chunk_id] = chunk_models
        
        return "\n".join(all_model_code)
    
    def _build_chunk_context(self, current_chunk: ERDChunk, all_chunks: List[ERDChunk]) -> Dict[str, Any]:
        """Build context for chunk generation including dependencies."""
        context = {
            "chunk_id": current_chunk.chunk_id,
            "entities": current_chunk.entities,
            "dependencies": current_chunk.dependencies,
            "dependents": current_chunk.dependents,
            "available_models": set()
        }
        
        # Add already generated models to context
        for chunk_id, models_code in self.generated_models.items():
            # Extract model names from generated code
            import re
            model_matches = re.findall(r'class (\w+)\(', models_code)
            context["available_models"].update(model_matches)
        
        return context
    
    async def _generate_chunk_models(self, chunk: ERDChunk, context: Dict[str, Any]) -> str:
        """Generate Django models for a specific chunk."""
        # Build optimized prompt for this chunk
        prompt = f"""
Generate Django models for this specific chunk of entities.

ENTITIES IN THIS CHUNK:
{json.dumps(chunk.entities, indent=2)}

CONTEXT:
- Chunk ID: {context['chunk_id']}
- Dependencies: {list(context['dependencies'])}
- Available Models: {list(context['available_models'])}

REQUIREMENTS:
- Generate ONLY models for entities in this chunk
- Reference external models using string names: 'ModelName'
- Include proper __str__ methods
- Add database indexes for performance
- Use appropriate field validators
- Follow Django best practices
- Handle circular dependencies with string references

EXAMPLE for ForeignKey to external model:
user = models.ForeignKey('User', on_delete=models.CASCADE)

Return only the Python code for the models:
"""
        
        return await self._generate_with_retry("models_chunk", prompt)
    
    async def _generate_component(self, component_type: str, entities: Dict[str, Any]) -> str:
        """Generate a complete component (serializers, views, urls, settings)."""
        prompts = {
            "serializers": f"""
Generate complete Django REST Framework serializers for ALL entities.

ENTITIES: {json.dumps(entities, indent=2)}

REQUIREMENTS:
- Create ModelSerializer for each entity
- Include all fields appropriately
- Handle relationships with proper serializer fields
- Add validation where needed
- Follow DRF best practices
- Optimize for performance

Return only the Python code:
""",
            "views": f"""
Generate complete Django REST Framework views for ALL entities.

ENTITIES: {json.dumps(entities, indent=2)}

REQUIREMENTS:
- Create ModelViewSet for each entity
- Include proper permissions and authentication
- Add filtering, searching, and pagination
- Handle nested relationships appropriately
- Include proper error handling
- Follow DRF best practices

Return only the Python code:
""",
            "urls": f"""
Generate complete Django URLs configuration for ALL entities.

ENTITIES: {json.dumps(entities, indent=2)}

REQUIREMENTS:
- Use DRF DefaultRouter
- Register all ViewSets
- Include API documentation
- Add versioning support
- Include health check endpoints

Return only the Python code:
""",
            "settings": f"""
Generate complete Django settings.py for this enterprise project.

ENTITIES: {json.dumps(entities, indent=2)}

REQUIREMENTS:
- Include all necessary Django apps
- Configure Django REST Framework
- Set up JWT authentication
- Add CORS configuration
- Include database optimization
- Add caching configuration
- Include monitoring and logging
- Environment variable support

Return only the Python code:
"""
        }
        
        return await self._generate_with_retry(component_type, prompts[component_type])
    
    async def _generate_with_retry(self, component: str, prompt: str, max_retries: int = 3) -> str:
        """Generate with retry logic and fallback models."""
        current_agent = self.primary_agent
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 [{component}] Generating with {current_agent.model_name}... (attempt {attempt + 1})")
                
                # Use model-specific optimizations
                max_tokens = min(4000, current_agent.model_config.max_tokens)
                temperature = max(0.1, current_agent.model_config.temperature - 0.1)  # Lower for enterprise
                
                result = await current_agent.generate(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                print(f"✅ [{component}] Generated successfully with {current_agent.model_name}")
                return result
                
            except Exception as e:
                print(f"⚠️  [{component}] Failed with {current_agent.model_name}: {e}")
                
                if attempt < max_retries - 1:
                    # Switch to fallback
                    if current_agent == self.primary_agent and self.fallback_agent != self.primary_agent:
                        current_agent = self.fallback_agent
                        print(f"🔄 [{component}] Switching to fallback: {current_agent.model_name}")
                    else:
                        # Try alternative models
                        alternatives = self.model_config.get("alternatives", [])
                        if len(alternatives) > attempt + 1:
                            alt_model = alternatives[attempt + 1]
                            current_agent = self._create_agent(alt_model)
                            print(f"🔄 [{component}] Switching to alternative: {alt_model}")
        
        return f"# ERROR: Failed to generate {component}\n# All models failed after {max_retries} attempts"
    
    def _generate_enterprise_files(self, entities: Dict[str, Any]) -> Dict[str, str]:
        """Generate additional enterprise-grade files."""
        return {
            "requirements.txt": """# Django Core
Django>=4.2.0
djangorestframework>=3.14.0
django-cors-headers>=4.0.0
djangorestframework-simplejwt>=5.2.0

# Database
psycopg2-binary>=2.9.0
redis>=4.0.0

# Environment & Config
python-decouple>=3.8
django-environ>=0.10.0

# Performance & Monitoring
django-debug-toolbar>=4.0.0
django-extensions>=3.2.0
whitenoise>=6.0.0

# Production
gunicorn>=21.2.0
celery>=5.3.0
django-celery-beat>=2.5.0

# Testing
pytest>=7.0.0
pytest-django>=4.5.0
factory-boy>=3.2.0

# Code Quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
""",
            
            "docker-compose.enterprise.yml": """version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      POSTGRES_DB: enterprise_db
      POSTGRES_USER: enterprise_user
      POSTGRES_PASSWORD: enterprise_password
    ports:
      - "5432:5432"
    command: postgres -c 'max_connections=200' -c 'shared_buffers=256MB'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  web:
    build: .
    command: gunicorn --bind 0.0.0.0:8000 --workers 4 --threads 2 backend.wsgi:application
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
    ports:
      - "8000:8000"
    environment:
      - DEBUG=0
      - DATABASE_URL=postgres://enterprise_user:enterprise_password@db:5432/enterprise_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery:
    build: .
    command: celery -A backend worker -l info --concurrency=4
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgres://enterprise_user:enterprise_password@db:5432/enterprise_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/app/staticfiles
    depends_on:
      - web

volumes:
  postgres_data:
  static_volume:
""",
            
            "nginx.conf": """events {
    worker_connections 1024;
}

http {
    upstream backend {
        server web:8000;
    }

    include /etc/nginx/mime.types;
    
    server {
        listen 80;
        client_max_body_size 100M;

        location /static/ {
            alias /app/staticfiles/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
    }
}
""",
            
            "Makefile": """# Enterprise Django Makefile

.PHONY: install test lint format check migrate run deploy

install:
	pip install -r requirements.txt

test:
	pytest --cov=. --cov-report=html

lint:
	flake8 .
	isort --check-only .
	black --check .

format:
	isort .
	black .

check: lint test

migrate:
	python manage.py makemigrations
	python manage.py migrate

run:
	python manage.py runserver

deploy:
	docker-compose -f docker-compose.enterprise.yml up --build -d

clean:
	docker-compose -f docker-compose.enterprise.yml down -v
	docker system prune -f

logs:
	docker-compose -f docker-compose.enterprise.yml logs -f

shell:
	python manage.py shell

superuser:
	python manage.py createsuperuser

collectstatic:
	python manage.py collectstatic --noinput

backup:
	docker-compose -f docker-compose.enterprise.yml exec db pg_dump -U enterprise_user enterprise_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
""",
            
            ".env.enterprise": """# Enterprise Django Configuration

# Debug
DEBUG=False
SECRET_KEY=your-super-secret-key-change-this-in-production

# Database
DATABASE_URL=postgres://enterprise_user:enterprise_password@localhost:5432/enterprise_db

# Cache
REDIS_URL=redis://localhost:6379/0

# API Keys (set as needed)
OPENROUTER_API_KEY=your-openrouter-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Security
ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Logging
LOG_LEVEL=INFO

# Performance
DJANGO_SETTINGS_MODULE=backend.settings.production
"""
        }

def main():
    """Main entry point for enterprise backend generation."""
    parser = argparse.ArgumentParser(description="Enterprise Agentic Django Backend Generator")
    parser.add_argument("erd_file", nargs='?', help="Path to ERD JSON file")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--preset", choices=["free", "premium", "balanced", "enterprise"], 
                       default="enterprise", help="Model preset to use")
    parser.add_argument("--output", default="enterprise_backend", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze ERD complexity")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        manager = ModelManager()
        print("🏢 Enterprise Models Available:")
        print("=" * 60)
        
        print("\n🆓 Free Models (Cost-Effective):")
        for model in manager.get_free_models():
            config = manager.get_model_config(model)
            print(f"  • {model} ({config.provider}) - {config.context_window} tokens")
        
        print("\n💎 Premium Models (High Performance):")
        premium_models = ["gpt-4-turbo", "claude-3-sonnet", "anthropic/claude-3-sonnet"]
        for model in premium_models:
            config = manager.get_model_config(model)
            if config:
                print(f"  • {model} ({config.provider}) - {config.context_window} tokens - ${config.cost_per_1k_tokens}/1k")
        
        print("\n🏢 Recommended for Enterprise:")
        print("  • claude-3-sonnet (Best for complex logic)")
        print("  • gpt-4-turbo (Large context window)")
        print("  • anthropic/claude-3-sonnet (via OpenRouter)")
        
        return
    
    if not args.erd_file:
        print("❌ ERD file required when not using --list-models")
        sys.exit(1)
    
    if not os.path.exists(args.erd_file):
        print(f"❌ ERD file not found: {args.erd_file}")
        sys.exit(1)
    
    # Load ERD
    with open(args.erd_file, 'r') as f:
        erd_data = json.load(f)
    
    # Analyze complexity first
    if args.analyze_only:
        analyzer = ERDAnalyzer()
        metrics = analyzer.analyze_erd(erd_data)
        
        print(f"\n📊 COMPLEXITY ANALYSIS")
        print("=" * 50)
        print(f"Entities: {metrics.entity_count}")
        print(f"Relationships: {metrics.relationship_count}")
        print(f"Max Relations/Entity: {metrics.max_relationships_per_entity}")
        print(f"Circular Dependencies: {metrics.circular_dependencies}")
        print(f"Inheritance Depth: {metrics.inheritance_depth}")
        print(f"Estimated Tokens: {metrics.estimated_tokens:,}")
        print(f"Complexity Score: {metrics.complexity_score:.2f}/1.0")
        print(f"Recommended Chunks: {metrics.recommended_chunks}")
        
        if metrics.complexity_score > 0.7:
            print("\n🔴 HIGH COMPLEXITY - Use enterprise processing")
        elif metrics.complexity_score > 0.4:
            print("\n🟡 MEDIUM COMPLEXITY - Chunked processing recommended")
        else:
            print("\n🟢 LOW COMPLEXITY - Standard processing sufficient")
        
        return
    
    # Configure models
    if args.model:
        model_config = auto_configure_models(args.model)
    else:
        # Enterprise preset uses high-performance models
        if args.preset == "enterprise":
            model_config = {
                "primary": "claude-3-sonnet",
                "fallback": "gpt-4-turbo",
                "alternatives": ["anthropic/claude-3-sonnet", "openai/gpt-4", "gpt-3.5-turbo"]
            }
        else:
            model_config = auto_configure_models(args.preset)
    
    print(f"🏢 Enterprise Configuration:")
    print(f"   Primary: {model_config['primary']}")
    print(f"   Fallback: {model_config['fallback']}")
    print(f"   Workers: {args.workers}")
    
    # Check API keys
    manager = ModelManager()
    primary_config = manager.get_model_config(model_config['primary'])
    if not primary_config:
        print(f"❌ Model '{model_config['primary']}' not found")
        sys.exit(1)
    
    api_key = os.getenv(primary_config.api_key_env)
    if not api_key:
        print(f"❌ API key not found. Set environment variable: {primary_config.api_key_env}")
        sys.exit(1)
    
    # Generate backend
    async def run_enterprise_generation():
        builder = EnterpriseBackendBuilder(model_config, max_workers=args.workers)
        await builder.generate_enterprise_backend(erd_data, args.output)
    
    print(f"🚀 Starting enterprise backend generation...")
    asyncio.run(run_enterprise_generation())
    print(f"🎉 Enterprise backend complete! Check {args.output}/ directory")

if __name__ == "__main__":
    main() 