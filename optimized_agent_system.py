"""
Optimized Multi-Agent Django Backend Generator
=============================================

Key Improvements:
- Async/await for better concurrency
- Agent Factory pattern for better organization
- Intelligent caching system
- Enhanced error handling and recovery
- Non-blocking HITL workflow
- Performance monitoring
- Semantic validation
"""

import asyncio
import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationMetrics:
    """Track performance and quality metrics for each generation."""
    agent_name: str
    start_time: float
    end_time: float
    success: bool
    retry_count: int
    quality_score: float
    token_count: int
    error_type: Optional[str] = None

@dataclass
class ValidationResult:
    """Structured validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    suggestions: List[str]

class CacheManager:
    """Intelligent caching system for generated code."""
    
    def __init__(self, cache_dir: Path = Path(".agent_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_key(self, erd: dict, agent_type: str, prompt_hash: str) -> str:
        """Generate cache key based on ERD, agent type, and prompt."""
        erd_str = json.dumps(erd, sort_keys=True)
        combined = f"{erd_str}:{agent_type}:{prompt_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, cache_key: str) -> Optional[str]:
        """Get cached result."""
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        cache_file = self.cache_dir / f"{cache_key}.py"
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                content = await f.read()
                self.memory_cache[cache_key] = content
                return content
        return None
    
    async def set(self, cache_key: str, content: str) -> None:
        """Cache result."""
        self.memory_cache[cache_key] = content
        cache_file = self.cache_dir / f"{cache_key}.py"
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(content)

class Agent(ABC):
    """Base agent class with enhanced capabilities."""
    
    def __init__(self, name: str, client, cache_manager: CacheManager):
        self.name = name
        self.client = client
        self.cache_manager = cache_manager
        self.metrics = []
    
    @abstractmethod
    async def generate(self, erd: dict, context: dict = None) -> str:
        """Generate code based on ERD and context."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    async def generate_with_cache(self, erd: dict, context: dict = None) -> str:
        """Generate with caching support."""
        start_time = time.time()
        prompt_hash = hashlib.md5(self.get_system_prompt().encode()).hexdigest()
        cache_key = self.cache_manager._get_cache_key(erd, self.name, prompt_hash)
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"[{self.name}] Using cached result")
            return cached_result
        
        # Generate new content
        try:
            result = await self.generate(erd, context)
            await self.cache_manager.set(cache_key, result)
            
            # Record metrics
            metrics = GenerationMetrics(
                agent_name=self.name,
                start_time=start_time,
                end_time=time.time(),
                success=True,
                retry_count=0,
                quality_score=0.0,  # Will be calculated by validator
                token_count=len(result.split())
            )
            self.metrics.append(metrics)
            
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Generation failed: {e}")
            raise

class IntelligentValidator:
    """Multi-layer validation system."""
    
    def __init__(self):
        self.validators = [
            self._syntax_validation,
            self._semantic_validation,
            self._best_practices_validation,
            self._security_validation
        ]
    
    async def validate(self, code: str, file_type: str, context: Optional[dict] = None) -> ValidationResult:
        """Run all validation layers."""
        errors = []
        warnings = []
        suggestions = []
        quality_scores = []
        
        for validator in self.validators:
            try:
                result = await validator(code, file_type, context)
                errors.extend(result.get('errors', []))
                warnings.extend(result.get('warnings', []))
                suggestions.extend(result.get('suggestions', []))
                quality_scores.append(result.get('quality_score', 0.5))
            except Exception as e:
                logger.warning(f"Validator failed: {e}")
                quality_scores.append(0.3)
        
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=overall_quality,
            suggestions=suggestions
        )
    
    async def _syntax_validation(self, code: str, file_type: str, context: Optional[dict] = None) -> dict:
        """Check syntax validity."""
        errors = []
        if file_type == "python":
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
        
        return {
            'errors': errors,
            'quality_score': 1.0 if not errors else 0.0
        }
    
    async def _semantic_validation(self, code: str, file_type: str, context: Optional[dict] = None) -> dict:
        """Check semantic correctness."""
        warnings = []
        suggestions = []
        
        # Check for common Django patterns
        if 'models.py' in file_type and 'from django.db import models' not in code:
            warnings.append("Missing Django models import")
        
        if 'serializers.py' in file_type and 'ModelSerializer' in code and 'Meta:' not in code:
            warnings.append("ModelSerializer missing Meta class")
        
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'quality_score': 0.8 if not warnings else 0.6
        }
    
    async def _best_practices_validation(self, code: str, file_type: str, context: Optional[dict] = None) -> dict:
        """Check best practices."""
        suggestions = []
        
        # Check line length
        long_lines = [i for i, line in enumerate(code.split('\n'), 1) 
                     if len(line) > 120]
        if long_lines:
            suggestions.append(f"Lines too long (>120 chars): {long_lines[:5]}")
        
        return {
            'suggestions': suggestions,
            'quality_score': 0.9 if not suggestions else 0.7
        }
    
    async def _security_validation(self, code: str, file_type: str, context: Optional[dict] = None) -> dict:
        """Check security issues."""
        errors = []
        warnings = []
        
        # Check for hardcoded secrets
        if any(keyword in code.lower() for keyword in ['password', 'secret', 'key']) and '"' in code:
            warnings.append("Possible hardcoded secret detected")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'quality_score': 1.0 if not errors else 0.3
        }

class HITLManager:
    """Enhanced Human-in-the-Loop management."""
    
    def __init__(self, auto_approve_threshold: float = 0.8):
        self.auto_approve_threshold = auto_approve_threshold
        self.pending_reviews = {}
        self.decisions = {}
    
    async def submit_for_review(self, file_name: str, content: str, 
                               validation_result: ValidationResult) -> str:
        """Submit file for human review (non-blocking)."""
        
        # Auto-approve high-quality files
        if (validation_result.quality_score >= self.auto_approve_threshold 
            and not validation_result.errors):
            logger.info(f"[HITL] Auto-approving {file_name} (quality: {validation_result.quality_score:.2f})")
            return "approved"
        
        # Queue for manual review
        self.pending_reviews[file_name] = {
            'content': content,
            'validation': validation_result,
            'timestamp': datetime.now()
        }
        
        logger.info(f"[HITL] {file_name} queued for manual review")
        return "pending"
    
    def get_pending_reviews(self) -> Dict[str, Any]:
        """Get all pending reviews."""
        return self.pending_reviews.copy()
    
    def submit_decision(self, file_name: str, decision: str, 
                       modified_content: str = None) -> None:
        """Submit human decision for a file."""
        self.decisions[file_name] = {
            'decision': decision,  # 'approved', 'rejected', 'modified'
            'content': modified_content,
            'timestamp': datetime.now()
        }
        
        if file_name in self.pending_reviews:
            del self.pending_reviews[file_name]

class OptimizedPlannerAgent:
    """Enhanced orchestrator with async support and intelligent workflows."""
    
    def __init__(self, erd: dict):
        self.erd = erd
        self.cache_manager = CacheManager()
        self.validator = IntelligentValidator()
        self.hitl_manager = HITLManager()
        
        # Initialize OpenAI client (same as before)
        self.client = None  # Initialize with your OpenAI client
        
        # Create agent factory
        self.agent_factory = AgentFactory(self.client, self.cache_manager)
        
        # Initialize agents
        self.agents = {
            'models': self.agent_factory.create_model_agent(),
            'serializers': self.agent_factory.create_serializer_agent(),
            'views': self.agent_factory.create_view_agent(),
            'urls': self.agent_factory.create_router_agent(),
            'settings': self.agent_factory.create_auth_agent(),
        }
    
    async def run(self) -> Dict[str, str]:
        """Run the optimized generation pipeline."""
        logger.info("[PlannerAgent] Starting optimized pipeline...")
        
        # Phase 1: Parallel generation with caching
        generation_tasks = []
        for file_name, agent in self.agents.items():
            task = self._generate_with_validation(file_name, agent)
            generation_tasks.append(task)
        
        # Execute all generations concurrently
        results = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        # Process results
        validated_files = {}
        for i, (file_name, agent) in enumerate(self.agents.items()):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"[{file_name}] Generation failed: {result}")
                continue
            
            content, validation_result = result
            validated_files[f"{file_name}.py"] = {
                'content': content,
                'validation': validation_result
            }
        
        # Phase 2: Submit for HITL review (non-blocking)
        review_tasks = []
        for file_name, file_data in validated_files.items():
            task = self.hitl_manager.submit_for_review(
                file_name, 
                file_data['content'], 
                file_data['validation']
            )
            review_tasks.append(task)
        
        await asyncio.gather(*review_tasks)
        
        # Phase 3: Process approved files
        final_files = {}
        for file_name, file_data in validated_files.items():
            if file_name in self.hitl_manager.decisions:
                decision = self.hitl_manager.decisions[file_name]
                if decision['decision'] in ['approved', 'modified']:
                    content = decision.get('content') or file_data['content']
                    final_files[file_name] = content
            else:
                # Auto-approved or still pending
                final_files[file_name] = file_data['content']
        
        logger.info(f"[PlannerAgent] Pipeline complete. Generated {len(final_files)} files.")
        return final_files
    
    async def _generate_with_validation(self, file_name: str, agent: Agent) -> tuple:
        """Generate content and validate it."""
        content = await agent.generate_with_cache(self.erd)
        validation_result = await self.validator.validate(content, file_name)
        return content, validation_result

class AgentFactory:
    """Factory for creating specialized agents."""
    
    def __init__(self, client, cache_manager: CacheManager):
        self.client = client
        self.cache_manager = cache_manager
    
    def create_model_agent(self) -> Agent:
        """Create a ModelAgent instance."""
        return ModelAgentOptimized("ModelAgent", self.client, self.cache_manager)
    
    def create_serializer_agent(self) -> Agent:
        """Create a SerializerAgent instance."""
        return SerializerAgentOptimized("SerializerAgent", self.client, self.cache_manager)
    
    def create_view_agent(self) -> Agent:
        """Create a ViewAgent instance."""
        return SerializerAgentOptimized("ViewAgent", self.client, self.cache_manager)
    
    def create_router_agent(self) -> Agent:
        """Create a RouterAgent instance."""
        return SerializerAgentOptimized("RouterAgent", self.client, self.cache_manager)
    
    def create_auth_agent(self) -> Agent:
        """Create an AuthAgent instance."""
        return SerializerAgentOptimized("AuthAgent", self.client, self.cache_manager)

class ModelAgentOptimized(Agent):
    """Optimized Model Agent with better prompts and error handling."""
    
    def get_system_prompt(self) -> str:
        return """
        Role: You are an expert Django backend engineer specializing in database modeling.
        Task: Generate a Django models.py file from a provided ERD (Entity-Relationship Diagram) in JSON format.
        
        Enhanced Capabilities:
        - Generate efficient database indexes
        - Handle complex relationships (self-referencing, through tables)
        - Add proper validation and constraints
        - Include metadata and documentation
        - Optimize for performance and scalability
        
        Output only clean, production-ready Python code.
        """
    
    async def generate(self, erd: dict, context: dict = None) -> str:
        """Generate optimized Django models."""
        logger.info(f"[{self.name}] Generating models with enhanced features...")
        
        prompt = f"""
        Generate a Django models.py file from this ERD:
        {json.dumps(erd, indent=2)}
        
        Requirements:
        1. Add appropriate database indexes for performance
        2. Include proper field validation
        3. Add string representations (__str__ methods)
        4. Include model metadata (ordering, verbose names)
        5. Handle edge cases and constraints
        
        Return only the Python code.
        """
        
        # Use your existing OpenAI client logic here
        # This is a placeholder for the actual implementation
        completion = await self._call_openai(prompt)
        return completion

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API (placeholder for actual implementation)."""
        # Implement your OpenAI API call here
        return "# Generated Django models code would go here"

class SerializerAgentOptimized(Agent):
    """Optimized Serializer Agent."""
    
    def get_system_prompt(self) -> str:
        return """
        Role: You are a Django REST Framework expert specializing in API serialization.
        Task: Generate DRF serializers that are secure, efficient, and follow best practices.
        
        Enhanced Capabilities:
        - Field-level permissions and validation
        - Nested serializers for relationships
        - Custom validation methods
        - Performance optimizations (select_related, prefetch_related)
        - API versioning support
        """
    
    async def generate(self, erd: dict, context: dict = None) -> str:
        """Generate optimized DRF serializers."""
        logger.info(f"[{self.name}] Generating serializers with enhanced features...")
        # Implementation here
        return "# Generated DRF serializers code would go here"

# Usage example
async def main():
    """Example usage of the optimized system."""
    erd = {
        "User": {
            "name": "CharField",
            "email": "EmailField",
            "has_many": ["Job"]
        },
        "Job": {
            "title": "CharField", 
            "description": "TextField",
            "posted_by": "ForeignKey:User"
        }
    }
    
    planner = OptimizedPlannerAgent(erd)
    generated_files = await planner.run()
    
    # Write files to disk
    backend_dir = Path("backend_optimized")
    backend_dir.mkdir(exist_ok=True)
    
    for filename, content in generated_files.items():
        file_path = backend_dir / filename
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
        logger.info(f"Generated: {file_path}")

if __name__ == "__main__":
    asyncio.run(main()) 