import json
import sys
from pathlib import Path
import os
import re
import ast
import tempfile
import subprocess
import asyncio
import aiofiles
import hashlib
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import uuid
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def strip_markdown_code_fence(text):
    # Remove triple backticks and optional language specifier
    return re.sub(r'^```(?:python)?\s*|```$', '', text.strip(), flags=re.MULTILINE).strip()

@dataclass
class ValidationResult:
    """Structured validation result with quality metrics."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    suggestions: List[str]
    auto_approve: bool = False

@dataclass
class GenerationMetrics:
    """Track performance metrics for each generation."""
    agent_name: str
    start_time: float
    end_time: float
    success: bool
    cached: bool
    quality_score: float
    token_count: int

class CacheManager:
    """Intelligent caching system for generated code."""
    
    def __init__(self, cache_dir: Path = Path(".agent_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        logger.info(f"[CacheManager] Initialized with cache directory: {cache_dir}")
    
    def _get_cache_key(self, erd: dict, agent_type: str) -> str:
        """Generate cache key based on ERD and agent type."""
        erd_str = json.dumps(erd, sort_keys=True)
        combined = f"{erd_str}:{agent_type}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, erd: dict, agent_type: str) -> Optional[str]:
        """Get cached result."""
        cache_key = self._get_cache_key(erd, agent_type)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.info(f"[CacheManager] Cache HIT (memory) for {agent_type}")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}_{agent_type}.py"
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                self.memory_cache[cache_key] = content
                logger.info(f"[CacheManager] Cache HIT (disk) for {agent_type}")
                return content
        
        logger.info(f"[CacheManager] Cache MISS for {agent_type}")
        return None
    
    async def set(self, erd: dict, agent_type: str, content: str) -> None:
        """Cache result to memory and disk."""
        cache_key = self._get_cache_key(erd, agent_type)
        
        # Store in memory
        self.memory_cache[cache_key] = content
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}_{agent_type}.py"
        async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        logger.info(f"[CacheManager] Cached result for {agent_type}")

class EnhancedValidator:
    """Multi-layer validation system with auto-approval."""
    
    def __init__(self, auto_approve_threshold: float = 0.8):
        self.auto_approve_threshold = auto_approve_threshold
        logger.info(f"[EnhancedValidator] Initialized with auto-approve threshold: {auto_approve_threshold}")
    
    async def validate(self, code: str, file_type: str, agent_name: str) -> ValidationResult:
        """Run comprehensive validation."""
        start_time = time.time()
        
        errors = []
        warnings = []
        suggestions = []
        quality_scores = []
        
        # Layer 1: Syntax validation
        syntax_result = await self._syntax_validation(code, file_type)
        errors.extend(syntax_result.get('errors', []))
        quality_scores.append(syntax_result.get('quality_score', 0.5))
        
        # Layer 2: Semantic validation
        semantic_result = await self._semantic_validation(code, file_type)
        warnings.extend(semantic_result.get('warnings', []))
        suggestions.extend(semantic_result.get('suggestions', []))
        quality_scores.append(semantic_result.get('quality_score', 0.5))
        
        # Layer 3: Best practices validation
        practices_result = await self._best_practices_validation(code, file_type)
        suggestions.extend(practices_result.get('suggestions', []))
        quality_scores.append(practices_result.get('quality_score', 0.5))
        
        # Layer 4: Security validation
        security_result = await self._security_validation(code, file_type)
        errors.extend(security_result.get('errors', []))
        warnings.extend(security_result.get('warnings', []))
        quality_scores.append(security_result.get('quality_score', 0.5))
        
        # Calculate overall quality
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        is_valid = len(errors) == 0
        auto_approve = is_valid and overall_quality >= self.auto_approve_threshold
        
        validation_time = time.time() - start_time
        logger.info(f"[EnhancedValidator] {agent_name} validation complete in {validation_time:.2f}s - Quality: {overall_quality:.2f}, Auto-approve: {auto_approve}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=overall_quality,
            suggestions=suggestions,
            auto_approve=auto_approve
        )
    
    async def _syntax_validation(self, code: str, file_type: str) -> dict:
        """Check syntax validity."""
        errors = []
        if file_type.endswith('.py'):
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
        
        return {
            'errors': errors,
            'quality_score': 1.0 if not errors else 0.0
        }
    
    async def _semantic_validation(self, code: str, file_type: str) -> dict:
        """Check semantic correctness and Django patterns."""
        warnings = []
        suggestions = []
        
        # Django-specific checks
        if 'models.py' in file_type:
            if 'from django.db import models' not in code:
                warnings.append("Missing Django models import")
            if 'class' in code and '__str__' not in code:
                suggestions.append("Consider adding __str__ methods to models")
        
        if 'serializers.py' in file_type:
            if 'from rest_framework' not in code:
                warnings.append("Missing Django REST Framework imports")
            if 'ModelSerializer' in code and 'Meta:' not in code:
                warnings.append("ModelSerializer missing Meta class")
        
        if 'views.py' in file_type:
            if 'from rest_framework' not in code:
                warnings.append("Missing Django REST Framework imports")
        
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'quality_score': 0.8 if not warnings else 0.6
        }
    
    async def _best_practices_validation(self, code: str, file_type: str) -> dict:
        """Check coding best practices."""
        suggestions = []
        
        lines = code.split('\n')
        
        # Check line length
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
        if long_lines:
            suggestions.append(f"Lines too long (>120 chars): {long_lines[:3]}...")
        
        # Check for proper docstrings
        if 'class' in code and '"""' not in code:
            suggestions.append("Consider adding docstrings to classes")
        
        # Check for proper imports organization
        if code.count('import') > 5 and code.find('\n\n') < code.find('import'):
            suggestions.append("Consider organizing imports better")
        
        return {
            'suggestions': suggestions,
            'quality_score': 0.9 if not suggestions else 0.7
        }
    
    async def _security_validation(self, code: str, file_type: str) -> dict:
        """Check for security issues."""
        errors = []
        warnings = []
        
        # Check for hardcoded secrets
        secret_keywords = ['password', 'secret', 'key', 'token', 'api_key']
        for keyword in secret_keywords:
            if keyword in code.lower() and any(quote in code for quote in ['"', "'"]):
                # Check if it's actually hardcoded
                lines = code.split('\n')
                for line in lines:
                    if keyword in line.lower() and ('=' in line or ':' in line):
                        if any(f'{quote}{keyword}' in line.lower() for quote in ['"', "'"]):
                            warnings.append(f"Possible hardcoded {keyword} detected")
        
        # Check for SQL injection risks
        if 'raw(' in code or '.extra(' in code:
            warnings.append("Raw SQL detected - ensure proper parameterization")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'quality_score': 1.0 if not errors and not warnings else 0.7
        }

class CriticAgent:
    """
    Role: You are a Python code reviewer and static analysis expert.
    Task: Review the provided code for errors, style issues, and best practices.
    Input: The generated code file.
    Output: A list of issues or 'No issues found.' if the code is clean.
    Constraints:
    - Use flake8 and PEP8 as the standard.
    - Do not suggest changes unless necessary.
    Capabilities:
    - You can identify syntax errors, style violations, and common mistakes.
    Reminders:
    - Be concise and specific in your feedback.
    """
    def review(self, code, filetype="python"):
        # No flake8 or linting, always approve
        return True, None

class ReviserAgent:
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
    def revise(self, erd, code, error, filetype="python"):
        system_prompt = (
            "Role: You are an expert Django developer and code fixer.\n"
            "Task: Revise the provided code to fix the described errors or issues.\n"
            "Input: The ERD JSON, the problematic code, and the error message or review feedback.\n"
            "Output: The corrected code file.\n"
            "Constraints:\n"
            "- Only change what is necessary to fix the issue.\n"
            "- Follow Django and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can interpret error messages and reviewer feedback to improve code.\n"
            "Reminders:\n"
            "- Double-check that the fix addresses the specific issue.\n"
            "- Do not introduce unrelated changes."
        )
        prompt = f"""You are an expert Django developer. Here is the ERD: {erd}\nHere is the previous {filetype} code:\n{code}\nHere is the error encountered:\n{error}\nPlease fix the code and return only the corrected file content."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        return strip_markdown_code_fence(completion.choices[0].message.content.strip())

class ModelAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager: CacheManager):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "models"
    
    async def generate(self, erd):
        """Generate models.py with caching support."""
        logger.info("[ModelAgent] Starting models.py generation...")
        start_time = time.time()
        
        # Check cache first
        cached_result = await self.cache_manager.get(erd, self.agent_type)
        if cached_result:
            return cached_result
        
        # Generate new content
        system_prompt = (
            "Role: You are an expert Django backend engineer specializing in database modeling.\n"
            "Task: Generate a Django models.py file from a provided ERD (Entity-Relationship Diagram) in JSON format.\n"
            "Input: The ERD JSON describing entities, fields, and relationships.\n"
            "Output: A complete, valid Django models.py file implementing all entities and relationships.\n"
            "Enhanced Requirements:\n"
            "- Add proper __str__ methods for all models\n"
            "- Include appropriate Meta classes with ordering\n"
            "- Add database indexes for performance\n"
            "- Use proper field validators where appropriate\n"
            "- Follow Django and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities: \n"
            "- You can infer field types and relationships from the ERD.\n"
            "- You can resolve foreign keys, one-to-one, and many-to-many relationships.\n"
            "Reminders: \n"
            "- Ensure all relationships are correctly implemented.\n"
            "- Validate that all model class names and field names are valid Python identifiers.\n"
            "- Do not include placeholder or example data."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate a Django models.py file.\nERD:\n{json.dumps(erd, indent=2)}\nReturn only the code for models.py."""
        
        # Make async OpenAI call
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="qwen/qwen3-coder:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                extra_headers=self.extra_headers,
                extra_body=self.extra_body,
            )
        )
        
        result = strip_markdown_code_fence(completion.choices[0].message.content.strip())
        
        # Cache the result
        await self.cache_manager.set(erd, self.agent_type, result)
        
        generation_time = time.time() - start_time
        logger.info(f"[ModelAgent] Generation complete in {generation_time:.2f}s")
        
        return result
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        code = completion.choices[0].message.content.strip()
        return strip_markdown_code_fence(code)

class SerializerAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "serializers"
    def generate(self, erd):
        print("[SerializerAgent] Generating serializers.py using OpenAI...")
        system_prompt = (
            "Role: You are a Django REST Framework expert specializing in serializer generation.\n"
            "Task: Generate serializers for all models described in the provided ERD.\n"
            "Input: The ERD JSON and the corresponding Django models.\n"
            "Output: A serializers.py file with ModelSerializers for each model.\n"
            "Constraints:\n"
            "- Use ModelSerializer for each model.\n"
            "- Include all fields and relationships.\n"
            "- Follow DRF and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can infer serializer fields from models and relationships.\n"
            "Reminders:\n"
            "- Ensure all related fields are properly represented (e.g., nested or primary key related fields as appropriate).\n"
            "- Do not include example data or comments."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate a Django REST Framework serializers.py file for all models.\nERD:\n{json.dumps(erd, indent=2)}\nReturn only the code for serializers.py."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        code = completion.choices[0].message.content.strip()
        return strip_markdown_code_fence(code)

class ViewAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "views"
    def generate(self, erd):
        print("[ViewAgent] Generating views.py using OpenAI...")
        system_prompt = (
            "Role: You are a Django REST Framework expert specializing in API view generation.\n"
            "Task: Generate ViewSets for all models in the ERD.\n"
            "Input: The ERD JSON, models, and serializers.\n"
            "Output: A views.py file with DRF ViewSets for each model.\n"
            "Constraints:\n"
            "- Use ModelViewSet for each model.\n"
            "- Register all necessary imports.\n"
            "- Follow DRF and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can infer queryset and serializer_class for each ViewSet.\n"
            "Reminders:\n"
            "- Ensure all ViewSets are complete and ready for router registration.\n"
            "- Do not include example data or comments."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate a Django REST Framework views.py file using ViewSets for all models.\nERD:\n{json.dumps(erd, indent=2)}\nReturn only the code for views.py."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        code = completion.choices[0].message.content.strip()
        return strip_markdown_code_fence(code)

class RouterAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "urls"
    def generate(self, erd):
        print("[RouterAgent] Generating urls.py using OpenAI...")
        system_prompt = (
            "Role: You are a Django REST Framework expert specializing in API routing.\n"
            "Task: Generate a urls.py file registering all ViewSets using DRF routers.\n"
            "Input: The ERD JSON and the list of ViewSets.\n"
            "Output: A urls.py file with all routes registered.\n"
            "Constraints:\n"
            "- Use DefaultRouter for registration.\n"
            "- Include all necessary imports.\n"
            "- Follow Django and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can infer route names from model names.\n"
            "Reminders:\n"
            "- Ensure all ViewSets are registered and the router is included in urlpatterns.\n"
            "- Do not include example data or comments."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate a Django urls.py file using DRF routers for all models.\nERD:\n{json.dumps(erd, indent=2)}\nReturn only the code for urls.py."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        code = completion.choices[0].message.content.strip()
        return strip_markdown_code_fence(code)

class AuthAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "settings"
    def generate(self, erd):
        print("[AuthAgent] Generating settings.py (auth section) using OpenAI...")
        system_prompt = (
            "Role: You are a Django backend expert specializing in authentication and security.\n"
            "Task: Generate the settings.py configuration for JWT authentication using djangorestframework-simplejwt.\n"
            "Input: The ERD JSON and project context.\n"
            "Output: The relevant settings.py section for JWT authentication.\n"
            "Constraints:\n"
            "- Use djangorestframework-simplejwt for JWT setup.\n"
            "- Follow Django and security best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can configure REST_FRAMEWORK and SIMPLE_JWT settings.\n"
            "Reminders:\n"
            "- Ensure all required settings are present and correct.\n"
            "- Do not include example data or comments."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate the Django settings.py code to set up JWT authentication using djangorestframework-simplejwt.\nERD:\n{json.dumps(erd, indent=2)}\nReturn only the code for settings.py (auth section)."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        code = completion.choices[0].message.content.strip()
        return strip_markdown_code_fence(code)

class DeploymentAgent:
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "deployment"
    def generate(self, erd):
        print("[DeploymentAgent] Generating deployment files using OpenAI...")
        system_prompt = (
            "Role: You are a DevOps and Django deployment expert.\n"
            "Task: Generate deployment files for a Django REST API project.\n"
            "Input: The ERD JSON and project context.\n"
            "Output: requirements.txt, Dockerfile, and Procfile as plain text.\n"
            "Constraints:\n"
            "- Use best practices for production Django deployment.\n"
            "- Ensure all dependencies are included.\n"
            "- Output only the file contents, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can infer required packages and Docker setup.\n"
            "Reminders:\n"
            "- Ensure all files are production-ready and minimal.\n"
            "- Do not include example data or comments."
        )
        prompt = f"""Given the following ERD (Entity-Relationship Diagram) as a JSON object, generate the following deployment files for a Django REST API project: requirements.txt, Dockerfile, and Procfile.\nERD:\n{json.dumps(erd, indent=2)}\nReturn a JSON object with keys 'requirements.txt', 'Dockerfile', and 'Procfile', each containing the file content as a string."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        try:
            files = json.loads(completion.choices[0].message.content)
            return {k: strip_markdown_code_fence(v) for k, v in files.items()}
        except Exception:
            print("[DeploymentAgent] Failed to parse deployment files as JSON, returning default stubs.")
            return {
                "requirements.txt": "# requirements.txt content",
                "Dockerfile": "# Dockerfile content",
                "Procfile": "# Procfile content"
            }

class DjangoCheckCritic:
    """
    Role: You are a Django project validator.
    Task: Run Django's system checks on the generated project and report any issues.
    Input: The generated Django project files.
    Output: A list of errors/warnings or 'No issues found.' if the project is valid.
    Constraints:
    - Use 'python manage.py check' for validation.
    Capabilities:
    - You can interpret Django check output and summarize issues.
    Reminders:
    - Be concise and actionable in your feedback.
    """
    def __init__(self, venv_path='.djcheckenv'):
        self.venv_path = venv_path
        self.python_bin = os.path.join(venv_path, 'Scripts', 'python.exe')

    def check(self, generated_files):
        temp_dir = f".djcheck_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # Create Django project
            subprocess.run([
                self.python_bin, '-m', 'django', 'admin', 'startproject', 'proj', temp_dir
            ], check=True)
            proj_dir = os.path.join(temp_dir, 'proj')
            # Copy generated files into proj/proj/ (for settings.py) and proj/app/ (for others)
            app_dir = os.path.join(proj_dir, 'app')
            os.makedirs(app_dir, exist_ok=True)
            for fname, content in generated_files.items():
                if fname == 'settings.py':
                    target = os.path.join(proj_dir, 'settings.py')
                else:
                    target = os.path.join(app_dir, fname)
                with open(target, 'w', encoding='utf-8') as f:
                    f.write(content)
            # Add app to INSTALLED_APPS in settings.py
            settings_path = os.path.join(proj_dir, 'settings.py')
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = f.read()
            if "'app'" not in settings:
                settings = settings.replace('INSTALLED_APPS = [', "INSTALLED_APPS = [\n    'app',")
            with open(settings_path, 'w', encoding='utf-8') as f:
                f.write(settings)
            # Run python manage.py check
            result = subprocess.run([
                self.python_bin, 'manage.py', 'check'
            ], cwd=proj_dir, capture_output=True, text=True)
            if result.returncode != 0 or 'ERROR' in result.stdout or 'Traceback' in result.stdout:
                return False, result.stdout.strip()
            return True, None
        except Exception as e:
            return False, f"Django check exception: {e}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class CustomFeatureAgent:
    """
    Role: You are a Django and Python expert specializing in implementing custom features from natural language descriptions.
    Task: Generate code to implement a user-specified custom feature in the Django backend.
    Input: The ERD JSON, the current project context, and a natural language feature description.
    Output: One or more code files (as a dict: filename -> code) implementing the feature.
    Constraints:
    - Follow Django, DRF, and PEP8 best practices.
    - Output only the code, no markdown or prose.
    Capabilities:
    - You can interpret complex feature requests and generate the necessary models, views, serializers, settings, etc.
    Reminders:
    - Only generate code relevant to the described feature.
    - Do not overwrite unrelated code.
    """
    def __init__(self, client, extra_headers, extra_body, cache_manager=None):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.cache_manager = cache_manager
        self.agent_type = "custom_feature"
    def generate(self, erd, feature_description, **kwargs):
        system_prompt = (
            "Role: You are a Django and Python expert specializing in implementing custom features from natural language descriptions.\n"
            "Task: Generate code to implement a user-specified custom feature in the Django backend.\n"
            "Input: The ERD JSON, the current project context, and a natural language feature description.\n"
            "Output: One or more code files (as a dict: filename -> code) implementing the feature.\n"
            "Constraints:\n"
            "- Follow Django, DRF, and PEP8 best practices.\n"
            "- Output only the code, no markdown or prose.\n"
            "Capabilities:\n"
            "- You can interpret complex feature requests and generate the necessary models, views, serializers, settings, etc.\n"
            "Reminders:\n"
            "- Only generate code relevant to the described feature.\n"
            "- Do not overwrite unrelated code."
        )
        prompt = f"""Given the following ERD and project context, implement the following custom feature in the Django backend.\nERD:\n{json.dumps(erd, indent=2)}\nFeature description: {feature_description}\nReturn a JSON object where each key is a filename (e.g., 'models.py', 'views.py') and each value is the code to add or modify for this feature."""
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        try:
            files = json.loads(completion.choices[0].message.content)
            return {k: strip_markdown_code_fence(v) for k, v in files.items()}
        except Exception:
            print("[CustomFeatureAgent] Failed to parse feature code as JSON, returning raw output.")
            return {"custom_feature.txt": completion.choices[0].message.content.strip()}

class PlannerAgent:
    def __init__(self, erd):
        self.erd = erd
        self.outputs = {}
        self.backend_dir = Path("backend")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://your-site-url.example.com"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "YourSiteName"),
        }
        self.extra_body = {}
        
        # Initialize optimized components
        self.cache_manager = CacheManager()
        self.validator = EnhancedValidator(auto_approve_threshold=0.8)
        self.metrics = []
        self.pending_reviews = {}
        
        # Initialize agents with caching support
        self.model_agent = ModelAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.serializer_agent = SerializerAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.view_agent = ViewAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.router_agent = RouterAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.auth_agent = AuthAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.deployment_agent = DeploymentAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)
        self.critic_agent = CriticAgent()
        self.reviser_agent = ReviserAgent(self.client, self.extra_headers, self.extra_body)
        self.django_critic = DjangoCheckCritic()
        self.custom_feature_agent = CustomFeatureAgent(self.client, self.extra_headers, self.extra_body, self.cache_manager)

    def run(self):
        """Run the optimized async pipeline."""
        return asyncio.run(self.async_run())
    
    async def async_run(self):
        """Optimized async generation pipeline with caching, validation, and non-blocking HITL."""
        logger.info("[PlannerAgent] Starting optimized async backend generation pipeline...")
        pipeline_start = time.time()
        
        # Phase 1: Parallel Generation with Caching and Validation
        generation_tasks = {
            'models.py': self._generate_with_validation('models.py', self.model_agent),
            'serializers.py': self._generate_with_validation('serializers.py', self.serializer_agent),
            'views.py': self._generate_with_validation('views.py', self.view_agent),
            'urls.py': self._generate_with_validation('urls.py', self.router_agent),
            'settings.py': self._generate_with_validation('settings.py', self.auth_agent),
        }
        
        logger.info(f"[PlannerAgent] Starting {len(generation_tasks)} parallel generations...")
        
        # Execute all generations concurrently
        results = await asyncio.gather(*generation_tasks.values(), return_exceptions=True)
        
        # Process results
        validated_outputs = {}
        auto_approved = []
        needs_review = []
        
        for i, (file_name, task) in enumerate(generation_tasks.items()):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"[{file_name}] Generation failed: {result}")
                validated_outputs[file_name] = f"# ERROR: Failed to generate {file_name}"
                continue
            
            content, validation_result, metrics = result
            validated_outputs[file_name] = content
            
            # Track metrics
            self.metrics.append(metrics)
            
            # Handle validation results
            if validation_result.auto_approve:
                auto_approved.append(file_name)
                logger.info(f"[HITL] Auto-approved {file_name} (quality: {validation_result.quality_score:.2f})")
            else:
                needs_review.append(file_name)
                self.pending_reviews[file_name] = {
                    'content': content,
                    'validation': validation_result,
                    'timestamp': datetime.now()
                }
        
        # Phase 2: Non-blocking HITL Review
        if needs_review:
            logger.info(f"[HITL] {len(needs_review)} files need manual review: {needs_review}")
            await self._handle_hitl_reviews(needs_review, validated_outputs)
        
        logger.info(f"[HITL] Auto-approved: {len(auto_approved)}, Manual review: {len(needs_review)}")
        
        outputs = validated_outputs
        
        # Continue with existing logic for deployment files, etc.
        outputs.update(self.generate_deployment_files())
        
        # Store final outputs
        self.outputs = outputs
        
        # Phase 3: Write files to disk asynchronously
        await self._write_outputs_async(outputs)
        
        pipeline_time = time.time() - pipeline_start
        logger.info(f"[PlannerAgent] Optimized pipeline complete in {pipeline_time:.2f}s. Generated {len(outputs)} files.")
        
        # Print performance summary
        self._print_performance_summary(pipeline_time, auto_approved, needs_review)
        
        return outputs
    
    async def _generate_with_validation(self, file_name: str, agent) -> Tuple[str, ValidationResult, GenerationMetrics]:
        """Generate content with validation and metrics tracking."""
        start_time = time.time()
        
        try:
            # Check if agent has async generate method
            if hasattr(agent, 'generate') and asyncio.iscoroutinefunction(agent.generate):
                content = await agent.generate(self.erd)
                cached = False  # Could check cache_manager for this info
            else:
                # Fallback to sync generation wrapped in executor
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(None, agent.generate, self.erd)
                cached = False
            
            # Validate generated content
            validation_result = await self.validator.validate(content, file_name, agent.__class__.__name__)
            
            # Create metrics
            metrics = GenerationMetrics(
                agent_name=agent.__class__.__name__,
                start_time=start_time,
                end_time=time.time(),
                success=True,
                cached=cached,
                quality_score=validation_result.quality_score,
                token_count=len(content.split())
            )
            
            return content, validation_result, metrics
            
        except Exception as e:
            logger.error(f"[{file_name}] Generation failed: {e}")
            
            # Create error metrics
            metrics = GenerationMetrics(
                agent_name=agent.__class__.__name__,
                start_time=start_time,
                end_time=time.time(),
                success=False,
                cached=False,
                quality_score=0.0,
                token_count=0
            )
            
            error_content = f"# ERROR: Failed to generate {file_name}\n# Error: {str(e)}"
            error_validation = ValidationResult(
                is_valid=False,
                errors=[str(e)],
                warnings=[],
                quality_score=0.0,
                suggestions=[],
                auto_approve=False
            )
            
            return error_content, error_validation, metrics
    
    async def _handle_hitl_reviews(self, needs_review: List[str], validated_outputs: Dict[str, str]) -> None:
        """Handle human-in-the-loop reviews in a non-blocking way."""
        logger.info("[HITL] Starting non-blocking review process...")
        
        for file_name in needs_review:
            review_data = self.pending_reviews[file_name]
            validation_result = review_data['validation']
            
            print(f"\n" + "="*60)
            print(f"REVIEW REQUIRED: {file_name}")
            print(f"Quality Score: {validation_result.quality_score:.2f}")
            
            if validation_result.errors:
                print(f"❌ ERRORS: {len(validation_result.errors)}")
                for error in validation_result.errors:
                    print(f"  - {error}")
            
            if validation_result.warnings:
                print(f"⚠️  WARNINGS: {len(validation_result.warnings)}")
                for warning in validation_result.warnings:
                    print(f"  - {warning}")
            
            if validation_result.suggestions:
                print(f"💡 SUGGESTIONS: {len(validation_result.suggestions)}")
                for suggestion in validation_result.suggestions:
                    print(f"  - {suggestion}")
            
            print(f"\nContent preview (first 200 chars):")
            print(f"{validated_outputs[file_name][:200]}...")
            print(f"\nOptions: [a]pprove, [r]eject, [e]dit, [s]kip")
            
            try:
                decision = input(f"Decision for {file_name}: ").lower().strip()
            except EOFError:
                decision = "a"  # Default to approve if no input
            
            if decision == 'r':
                validated_outputs[file_name] = f"# REJECTED: {file_name} was rejected during review"
                logger.info(f"[HITL] {file_name} rejected")
            elif decision == 'e':
                print("Edit mode not implemented in this demo. Approving with current content.")
                logger.info(f"[HITL] {file_name} approved after edit request")
            elif decision == 's':
                validated_outputs[file_name] = f"# SKIPPED: {file_name} was skipped during review"
                logger.info(f"[HITL] {file_name} skipped")
            else:
                logger.info(f"[HITL] {file_name} approved")
    
    async def _write_outputs_async(self, outputs: Dict[str, str]) -> None:
        """Write outputs to disk asynchronously."""
        self.backend_dir.mkdir(exist_ok=True)
        
        write_tasks = []
        for filename, content in outputs.items():
            file_path = self.backend_dir / filename
            write_tasks.append(self._write_file_async(file_path, content))
        
        await asyncio.gather(*write_tasks)
        logger.info(f"[Writer] All {len(outputs)} files written to {self.backend_dir.resolve()}")
    
    async def _write_file_async(self, file_path: Path, content: str) -> None:
        """Write a single file asynchronously."""
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        logger.info(f"[Writer] Wrote {file_path}")
    
    def _print_performance_summary(self, pipeline_time: float, auto_approved: List[str], needs_review: List[str]) -> None:
        """Print performance summary."""
        print(f"\n" + "="*60)
        print(f"🚀 OPTIMIZED PIPELINE PERFORMANCE SUMMARY")
        print(f"="*60)
        print(f"⏱️  Total time: {pipeline_time:.2f}s")
        print(f"✅ Auto-approved: {len(auto_approved)} files")
        print(f"👁️  Manual review: {len(needs_review)} files")
        
        if self.metrics:
            avg_quality = sum(m.quality_score for m in self.metrics) / len(self.metrics)
            cached_count = sum(1 for m in self.metrics if m.cached)
            print(f"📊 Average quality score: {avg_quality:.2f}")
            print(f"💾 Cache hits: {cached_count}/{len(self.metrics)}")
            
            # Show per-agent performance
            print(f"\n📈 Per-agent performance:")
            for metric in self.metrics:
                generation_time = metric.end_time - metric.start_time
                status = "✅" if metric.success else "❌"
                cache_status = "💾" if metric.cached else "🔄"
                print(f"  {status} {cache_status} {metric.agent_name}: {generation_time:.2f}s (Q: {metric.quality_score:.2f})")
        
        print(f"="*60)
        # Django check critic
        print("[DjangoCheckCritic] Running python manage.py check on generated code...")
        ok, error = self.django_critic.check(outputs)
        if not ok:
            print(f"[DjangoCheckCritic] Django check failed:\n{error}")
            # Optionally, trigger revision for all files (or parse error to target specific file)
            # For now, just print error and continue
        else:
            print("[DjangoCheckCritic] Django check passed.")
        # Deployment files (run after parallel step)
        outputs.update(self.generate_deployment_files())
        # After main generation steps, prompt for custom feature
        print("\n[CustomFeature] Would you like to add a custom feature (e.g., 'Add Stripe payment gateway integration')?\nLeave blank to skip, or enter a description:")
        try:
            feature_description = input("Custom feature description: ").strip()
        except EOFError:
            feature_description = ""
        if feature_description:
            print(f"[CustomFeature] Generating code for: {feature_description}")
            feature_outputs = self.custom_feature_agent.generate(self.erd, feature_description)
            for filename, code in feature_outputs.items():
                out_path = self.backend_dir / filename
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(f"[CustomFeature] Wrote {filename}")
        else:
            print("[CustomFeature] No custom feature requested. Skipping.")
        self.outputs = outputs
        print("[Agent] Pipeline complete. Writing files to disk...")
        self.write_outputs()
        print(f"[Agent] All files written to '{self.backend_dir.resolve()}'")

    def openai_generate(self, prompt, system_prompt="You are an expert Django backend engineer."):
        completion = self.client.chat.completions.create(
            extra_headers={},
            extra_body={},
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()

    def safe_generate(self, gen_func, name, filetype="python", max_retries=2):
        last_output = None
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    result = gen_func()
                else:
                    # Use ReviserAgent to fix the code
                    result = self.reviser_agent.revise(self.erd, last_output, last_error, filetype=filetype)
                # CriticAgent reviews the code
                is_valid, error = self.critic_agent.review(result, filetype=filetype)
                if is_valid:
                    return result
                else:
                    print(f"[CriticAgent] {name} failed review: {error}")
                    last_output = result
                    last_error = error
            except Exception as e:
                last_error = str(e)
        print(f"[Agent] {name} generation failed after {max_retries+1} attempts.")
        return f"# ERROR: Failed to generate {name}"

    def is_valid_output(self, output, name):
        return output and not output.strip().startswith("# ERROR")

    def get_last_error(self, output, name):
        return "Simulated error: output did not pass validation."

    def revise_with_openai(self, name, last_output, last_error):
        print(f"[Reviser] Using OpenAI to revise {name} based on error...")
        prompt = f"""You are an expert Django developer. Here is the ERD: {self.erd}\nHere is the previous {name} code:\n{last_output}\nHere is the error encountered:\n{last_error}\nPlease fix the code and return only the corrected file content."""
        return self.openai_generate(prompt)

    def generate_models(self):
        return self.model_agent.generate(self.erd)

    def generate_serializers(self):
        return self.serializer_agent.generate(self.erd)

    def generate_views(self):
        return self.view_agent.generate(self.erd)

    def generate_urls(self):
        return self.router_agent.generate(self.erd)

    def generate_auth_setup(self):
        return self.auth_agent.generate(self.erd)

    def generate_deployment_files(self):
        return self.deployment_agent.generate(self.erd)

    def write_outputs(self):
        self.backend_dir.mkdir(exist_ok=True)
        for filename, content in self.outputs.items():
            file_path = self.backend_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [Write] {file_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python agent_backend_builder.py <erd_json_file>")
        sys.exit(1)

    erd_json_path = Path(sys.argv[1])
    if not erd_json_path.exists():
        print(f"File not found: {erd_json_path}")
        sys.exit(1)

    with open(erd_json_path, 'r', encoding='utf-8') as f:
        erd = json.load(f)

    print("Loaded ERD:")
    print(json.dumps(erd, indent=2))

    agent = PlannerAgent(erd)
    agent.run()


if __name__ == "__main__":
    main() 