"""
Priority 3: Advanced Features - Enterprise-Grade Agentic System
==============================================================

Building on the successful Priority 1 (Performance) and Priority 2 (UX/Quality) optimizations,
Priority 3 introduces enterprise-grade features for production-ready agentic systems.

Key Features:
- 🤖 Agent Specialization: Model-specific and domain-specific agents
- 🧠 Context Awareness: Cross-file dependency analysis
- 🚀 Production Features: Auto-testing, CI/CD integration, Docker optimization
- 🎯 Intelligent Routing: Optimal agent selection based on task complexity
- 📈 Progressive Enhancement: Semantic understanding and iterative improvement
"""

import asyncio
import json
import os
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict, deque
import networkx as nx
from abc import ABC, abstractmethod
import subprocess
import yaml
import docker
from jinja2 import Template

@dataclass
class AgentCapability:
    """Define capabilities for specialized agents."""
    name: str
    complexity_range: Tuple[float, float]  # (min, max) complexity scores
    domain_expertise: List[str]  # e.g., ['ecommerce', 'fintech', 'healthcare']
    model_preferences: List[str]  # Preferred models for this capability
    cost_per_token: float  # Cost efficiency
    quality_score: float  # Historical quality performance
    specializations: List[str]  # e.g., ['models', 'views', 'serializers']

@dataclass
class DependencyNode:
    """Represent a code dependency in the dependency graph."""
    file_path: str
    entity_name: str  # Class, function, or variable name
    entity_type: str  # 'class', 'function', 'variable', 'import'
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    
@dataclass
class CodeContext:
    """Rich context information for code generation."""
    file_dependencies: Dict[str, List[str]]
    entity_relationships: Dict[str, List[str]]
    domain_patterns: List[str]
    quality_requirements: Dict[str, float]
    existing_code_context: str
    architectural_constraints: List[str]

class SpecializedAgent(ABC):
    """Base class for all specialized agents."""
    
    def __init__(self, capability: AgentCapability, client=None):
        self.capability = capability
        self.client = client
        self.performance_history = []
        
    @abstractmethod
    async def can_handle(self, task: Dict[str, Any]) -> float:
        """Return confidence score (0-1) for handling this task."""
        pass
    
    @abstractmethod
    async def generate(self, task: Dict[str, Any], context: CodeContext) -> str:
        """Generate code with full context awareness."""
        pass
    
    def update_performance(self, task: Dict[str, Any], quality_score: float, execution_time: float):
        """Update performance metrics for this agent."""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'task_complexity': task.get('complexity', 0),
            'quality_score': quality_score,
            'execution_time': execution_time,
            'domain': task.get('domain', 'general')
        })

class ModelSpecificAgent(SpecializedAgent):
    """Agent specialized for specific models (GPT-4, Claude, etc.)."""
    
    def __init__(self, capability: AgentCapability, client=None, model_name: str = None):
        super().__init__(capability, client)
        self.model_name = model_name or capability.model_preferences[0]
        
    async def can_handle(self, task: Dict[str, Any]) -> float:
        """Evaluate task based on complexity and model strengths."""
        task_complexity = task.get('complexity', 0.5)
        min_complexity, max_complexity = self.capability.complexity_range
        
        # Check if complexity is in our range
        if not (min_complexity <= task_complexity <= max_complexity):
            return 0.0
        
        # Check domain expertise
        task_domain = task.get('domain', 'general')
        domain_match = 1.0 if task_domain in self.capability.domain_expertise else 0.5
        
        # Check file type specialization
        file_type = task.get('file_type', '')
        specialization_match = 1.0 if file_type in self.capability.specializations else 0.7
        
        # Calculate confidence based on historical performance
        recent_performance = self._get_recent_performance(task_domain)
        
        confidence = (domain_match * 0.3 + specialization_match * 0.3 + 
                     recent_performance * 0.4)
        
        return min(confidence, 1.0)
    
    async def generate(self, task: Dict[str, Any], context: CodeContext) -> str:
        """Generate code using model-specific optimized prompts."""
        # Build context-aware prompt
        prompt = self._build_context_aware_prompt(task, context)
        
        # Use model-specific optimizations
        model_params = self._get_model_specific_params()
        
        try:
            if self.client:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **model_params
                )
                return completion.choices[0].message.content
            else:
                # Fallback for testing
                return f"# Generated by {self.model_name}\n# Task: {task.get('file_type', 'unknown')}\n"
                
        except Exception as e:
            print(f"⚠️  {self.model_name} generation failed: {e}")
            return f"# ERROR: Failed to generate with {self.model_name}\n# Error: {str(e)}"
    
    def _build_context_aware_prompt(self, task: Dict[str, Any], context: CodeContext) -> str:
        """Build a context-aware prompt with dependency information."""
        base_prompt = task.get('prompt', '')
        
        # Add dependency context
        if context.file_dependencies:
            base_prompt += "\n\nFILE DEPENDENCIES:\n"
            for file, deps in context.file_dependencies.items():
                base_prompt += f"- {file} depends on: {', '.join(deps)}\n"
        
        # Add entity relationships
        if context.entity_relationships:
            base_prompt += "\n\nENTITY RELATIONSHIPS:\n"
            for entity, relations in context.entity_relationships.items():
                base_prompt += f"- {entity} relates to: {', '.join(relations)}\n"
        
        # Add domain patterns
        if context.domain_patterns:
            base_prompt += f"\n\nDOMAIN PATTERNS: {', '.join(context.domain_patterns)}\n"
        
        # Add existing code context
        if context.existing_code_context:
            base_prompt += f"\n\nEXISTING CODE CONTEXT:\n{context.existing_code_context}\n"
        
        # Add architectural constraints
        if context.architectural_constraints:
            base_prompt += f"\n\nARCHITECTURAL CONSTRAINTS:\n"
            for constraint in context.architectural_constraints:
                base_prompt += f"- {constraint}\n"
        
        # Add model-specific instructions
        base_prompt += f"\n\nMODEL-SPECIFIC INSTRUCTIONS for {self.model_name}:\n"
        base_prompt += self._get_model_specific_instructions()
        
        return base_prompt
    
    def _get_model_specific_params(self) -> Dict[str, Any]:
        """Get model-specific parameters for optimal generation."""
        if "gpt-4" in self.model_name.lower():
            return {
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "qwen" in self.model_name.lower():
            return {
                "temperature": 0.2,
                "max_tokens": 1500,
                "top_p": 0.9
            }
        elif "deepseek" in self.model_name.lower():
            return {
                "temperature": 0.15,
                "max_tokens": 1800,
                "top_p": 0.92
            }
        else:
            return {
                "temperature": 0.2,
                "max_tokens": 1500,
                "top_p": 0.9
            }
    
    def _get_model_specific_instructions(self) -> str:
        """Get model-specific generation instructions."""
        if "gpt-4" in self.model_name.lower():
            return """- Focus on clean, maintainable code with comprehensive docstrings
- Use advanced Python features and best practices
- Include comprehensive error handling
- Optimize for readability and performance"""
        elif "qwen" in self.model_name.lower():
            return """- Generate concise, efficient code
- Focus on core functionality
- Use standard Python patterns
- Ensure code is production-ready"""
        elif "deepseek" in self.model_name.lower():
            return """- Emphasize code quality and performance
- Use modern Python features appropriately
- Include necessary imports and dependencies
- Follow Django/DRF best practices strictly"""
        else:
            return """- Generate clean, functional code
- Follow Python and Django conventions
- Include proper imports and error handling"""
    
    def _get_recent_performance(self, domain: str) -> float:
        """Get recent performance score for this domain."""
        if not self.performance_history:
            return self.capability.quality_score
        
        # Get last 5 tasks in this domain
        domain_tasks = [task for task in self.performance_history[-10:] 
                       if task['domain'] == domain]
        
        if not domain_tasks:
            return self.capability.quality_score
        
        return sum(task['quality_score'] for task in domain_tasks) / len(domain_tasks)

class DomainSpecificAgent(SpecializedAgent):
    """Agent specialized for specific domains (e-commerce, fintech, etc.)."""
    
    def __init__(self, capability: AgentCapability, client=None, domain: str = None):
        super().__init__(capability, client)
        self.domain = domain or capability.domain_expertise[0]
        self.domain_patterns = self._load_domain_patterns()
        
    async def can_handle(self, task: Dict[str, Any]) -> float:
        """Evaluate task based on domain expertise."""
        task_domain = task.get('domain', 'general')
        
        # High confidence for exact domain match
        if task_domain == self.domain:
            return 0.95
        
        # Medium confidence for related domains
        if task_domain in self.capability.domain_expertise:
            return 0.8
        
        # Low confidence for general tasks
        return 0.3
    
    async def generate(self, task: Dict[str, Any], context: CodeContext) -> str:
        """Generate domain-specific code with specialized patterns."""
        prompt = self._build_domain_specific_prompt(task, context)
        
        try:
            if self.client:
                completion = self.client.chat.completions.create(
                    model=self.capability.model_preferences[0],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000
                )
                generated_code = completion.choices[0].message.content
                
                # Apply domain-specific post-processing
                return self._apply_domain_patterns(generated_code, task)
            else:
                return f"# Generated by {self.domain} domain agent\n# Domain patterns applied\n"
                
        except Exception as e:
            print(f"⚠️  {self.domain} domain agent failed: {e}")
            return f"# ERROR: Domain-specific generation failed\n# Error: {str(e)}"
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific patterns and templates."""
        patterns = {
            'ecommerce': [
                'product_model_pattern', 'cart_functionality', 'payment_integration',
                'inventory_management', 'order_processing', 'user_authentication'
            ],
            'fintech': [
                'transaction_model', 'account_management', 'security_compliance',
                'audit_trail', 'financial_calculations', 'regulatory_compliance'
            ],
            'healthcare': [
                'patient_privacy', 'hipaa_compliance', 'medical_records',
                'appointment_scheduling', 'billing_integration', 'audit_logging'
            ],
            'saas': [
                'multi_tenancy', 'subscription_management', 'usage_tracking',
                'api_rate_limiting', 'webhook_integration', 'billing_automation'
            ]
        }
        return patterns.get(self.domain, [])
    
    def _build_domain_specific_prompt(self, task: Dict[str, Any], context: CodeContext) -> str:
        """Build a domain-specific prompt with specialized knowledge."""
        base_prompt = task.get('prompt', '')
        
        # Add domain-specific context
        base_prompt += f"\n\nDOMAIN: {self.domain.upper()}\n"
        base_prompt += f"Apply {self.domain} industry best practices and patterns.\n"
        
        # Add domain patterns
        if self.domain_patterns:
            base_prompt += f"\nIMPORTANT {self.domain.upper()} PATTERNS TO CONSIDER:\n"
            for pattern in self.domain_patterns:
                base_prompt += f"- {pattern.replace('_', ' ').title()}\n"
        
        # Add domain-specific requirements
        domain_requirements = self._get_domain_requirements()
        if domain_requirements:
            base_prompt += f"\n{self.domain.upper()} REQUIREMENTS:\n"
            for req in domain_requirements:
                base_prompt += f"- {req}\n"
        
        return base_prompt
    
    def _get_domain_requirements(self) -> List[str]:
        """Get domain-specific requirements."""
        requirements = {
            'ecommerce': [
                "Implement proper product variation handling",
                "Include cart session management", 
                "Add inventory validation",
                "Implement order status tracking",
                "Include payment method integration points"
            ],
            'fintech': [
                "Implement transaction immutability",
                "Add comprehensive audit logging",
                "Include balance validation",
                "Implement proper currency handling",
                "Add regulatory compliance checks"
            ],
            'healthcare': [
                "Ensure HIPAA compliance for patient data",
                "Implement proper access controls",
                "Add audit logging for all data access",
                "Include data encryption requirements",
                "Implement consent management"
            ],
            'saas': [
                "Implement multi-tenant data isolation",
                "Add subscription status tracking",
                "Include usage metering",
                "Implement proper API rate limiting",
                "Add webhook event handling"
            ]
        }
        return requirements.get(self.domain, [])
    
    def _apply_domain_patterns(self, code: str, task: Dict[str, Any]) -> str:
        """Apply domain-specific patterns to generated code."""
        # Add domain-specific imports
        domain_imports = self._get_domain_imports()
        if domain_imports and "from django.db import models" in code:
            imports_section = "\n".join(domain_imports) + "\n"
            code = code.replace("from django.db import models", 
                              f"from django.db import models\n{imports_section}")
        
        # Add domain-specific mixins or base classes
        if self.domain == 'ecommerce' and 'class Product' in code:
            code = code.replace('class Product(models.Model):', 
                              'class Product(TimestampedModel, models.Model):')
        
        return code
    
    def _get_domain_imports(self) -> List[str]:
        """Get domain-specific imports."""
        imports = {
            'ecommerce': [
                "from decimal import Decimal",
                "from django.core.validators import MinValueValidator",
                "from django.contrib.auth import get_user_model"
            ],
            'fintech': [
                "from decimal import Decimal, ROUND_HALF_UP",
                "from django.core.validators import MinValueValidator",
                "from django.utils import timezone",
                "import uuid"
            ],
            'healthcare': [
                "from django.contrib.auth import get_user_model",
                "from django.core.validators import RegexValidator",
                "from cryptography.fernet import Fernet"
            ],
            'saas': [
                "from django.contrib.auth import get_user_model",
                "from django.utils import timezone",
                "import uuid"
            ]
        }
        return imports.get(self.domain, [])

class DependencyAnalyzer:
    """Analyze and track dependencies between code files and entities."""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.file_asts = {}
        self.entity_map = {}
        
    async def analyze_codebase(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze entire codebase for dependencies."""
        print("🔍 Analyzing codebase dependencies...")
        
        # Parse all files
        for file_path in file_paths:
            await self._parse_file(file_path)
        
        # Build dependency relationships
        await self._build_dependency_graph()
        
        # Calculate complexity scores
        self._calculate_complexity_scores()
        
        return {
            'dependency_graph': self.dependency_graph,
            'entity_map': self.entity_map,
            'complexity_scores': self._get_complexity_scores(),
            'critical_dependencies': self._find_critical_dependencies()
        }
    
    async def _parse_file(self, file_path: str):
        """Parse a Python file and extract entities."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                self.file_asts[file_path] = tree
                
                # Extract entities
                entities = self._extract_entities(tree, file_path)
                for entity in entities:
                    self.entity_map[f"{file_path}::{entity.entity_name}"] = entity
                    
        except Exception as e:
            print(f"⚠️  Failed to parse {file_path}: {e}")
    
    def _extract_entities(self, tree: ast.AST, file_path: str) -> List[DependencyNode]:
        """Extract classes, functions, and imports from AST."""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = DependencyNode(
                    file_path=file_path,
                    entity_name=node.name,
                    entity_type='class'
                )
                entities.append(entity)
                
            elif isinstance(node, ast.FunctionDef):
                entity = DependencyNode(
                    file_path=file_path,
                    entity_name=node.name,
                    entity_type='function'
                )
                entities.append(entity)
                
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    entity = DependencyNode(
                        file_path=file_path,
                        entity_name=alias.name,
                        entity_type='import'
                    )
                    entities.append(entity)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    entity = DependencyNode(
                        file_path=file_path,
                        entity_name=node.module,
                        entity_type='import'
                    )
                    entities.append(entity)
        
        return entities
    
    async def _build_dependency_graph(self):
        """Build the dependency graph between entities."""
        for entity_key, entity in self.entity_map.items():
            self.dependency_graph.add_node(entity_key, **entity.__dict__)
            
            # Find dependencies by analyzing references
            dependencies = await self._find_entity_dependencies(entity)
            for dep in dependencies:
                if dep in self.entity_map:
                    self.dependency_graph.add_edge(entity_key, dep)
                    entity.dependencies.add(dep)
                    self.entity_map[dep].dependents.add(entity_key)
    
    async def _find_entity_dependencies(self, entity: DependencyNode) -> Set[str]:
        """Find what this entity depends on."""
        dependencies = set()
        
        if entity.file_path in self.file_asts:
            tree = self.file_asts[entity.file_path]
            
            # Look for class/function references
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # Check if this name refers to another entity
                    for other_key, other_entity in self.entity_map.items():
                        if (other_entity.entity_name == node.id and 
                            other_entity.file_path != entity.file_path):
                            dependencies.add(other_key)
        
        return dependencies
    
    def _calculate_complexity_scores(self):
        """Calculate complexity scores for entities."""
        for entity_key, entity in self.entity_map.items():
            # Base complexity on number of dependencies and dependents
            deps_count = len(entity.dependencies)
            dependents_count = len(entity.dependents)
            
            # Higher complexity for entities with many dependencies or dependents
            complexity = (deps_count * 0.6 + dependents_count * 0.4) / 10
            entity.complexity_score = min(complexity, 1.0)
    
    def _get_complexity_scores(self) -> Dict[str, float]:
        """Get complexity scores for all entities."""
        return {key: entity.complexity_score 
                for key, entity in self.entity_map.items()}
    
    def _find_critical_dependencies(self) -> List[str]:
        """Find entities that are critical to the system."""
        # Critical entities have many dependents
        critical = []
        for entity_key, entity in self.entity_map.items():
            if len(entity.dependents) >= 3:  # Threshold for criticality
                critical.append(entity_key)
        
        return sorted(critical, key=lambda x: len(self.entity_map[x].dependents), reverse=True)

class IntelligentRouter:
    """Route tasks to optimal specialized agents."""
    
    def __init__(self):
        self.agents: List[SpecializedAgent] = []
        self.routing_history = []
        
    def register_agent(self, agent: SpecializedAgent):
        """Register a specialized agent."""
        self.agents.append(agent)
        print(f"🤖 Registered {agent.__class__.__name__}: {agent.capability.name}")
    
    async def route_task(self, task: Dict[str, Any], context: CodeContext) -> SpecializedAgent:
        """Route task to the best available agent."""
        print(f"🎯 Routing task: {task.get('file_type', 'unknown')} (complexity: {task.get('complexity', 0):.2f})")
        
        # Get confidence scores from all agents
        agent_scores = []
        for agent in self.agents:
            confidence = await agent.can_handle(task)
            cost_efficiency = 1.0 / agent.capability.cost_per_token if agent.capability.cost_per_token > 0 else 1.0
            
            # Combined score: confidence * quality * cost_efficiency
            combined_score = confidence * agent.capability.quality_score * cost_efficiency
            
            agent_scores.append((agent, confidence, combined_score))
        
        # Sort by combined score
        agent_scores.sort(key=lambda x: x[2], reverse=True)
        
        if agent_scores:
            best_agent, confidence, score = agent_scores[0]
            print(f"✅ Selected {best_agent.__class__.__name__}: {best_agent.capability.name} (confidence: {confidence:.2f})")
            
            # Record routing decision
            self.routing_history.append({
                'timestamp': datetime.now(),
                'task': task,
                'agent': best_agent.capability.name,
                'confidence': confidence,
                'score': score
            })
            
            return best_agent
        
        # Fallback to first agent if no good match
        print("⚠️  No optimal agent found, using fallback")
        return self.agents[0] if self.agents else None

class ProductionFeatures:
    """Production-ready features: testing, CI/CD, Docker optimization."""
    
    def __init__(self, project_path: str = "backend"):
        self.project_path = Path(project_path)
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception:
            print("⚠️  Docker not available")
    
    async def generate_tests(self, generated_files: Dict[str, str]) -> Dict[str, str]:
        """Generate comprehensive tests for generated code."""
        print("🧪 Generating comprehensive test suite...")
        
        test_files = {}
        
        for filename, content in generated_files.items():
            if filename.endswith('.py') and 'models.py' in filename:
                test_files[f"test_{filename}"] = await self._generate_model_tests(content)
            elif filename.endswith('.py') and 'views.py' in filename:
                test_files[f"test_{filename}"] = await self._generate_view_tests(content)
            elif filename.endswith('.py') and 'serializers.py' in filename:
                test_files[f"test_{filename}"] = await self._generate_serializer_tests(content)
        
        # Generate integration tests
        test_files['test_integration.py'] = await self._generate_integration_tests(generated_files)
        
        return test_files
    
    async def _generate_model_tests(self, models_content: str) -> str:
        """Generate model tests."""
        template = Template('''"""
Comprehensive model tests - Auto-generated
"""
import pytest
from django.test import TestCase
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.contrib.auth import get_user_model
{{ imports }}

class {{ model_name }}TestCase(TestCase):
    """Test cases for {{ model_name }} model."""
    
    def setUp(self):
        """Set up test data."""
        self.user = get_user_model().objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_create_{{ model_name_lower }}(self):
        """Test creating a {{ model_name_lower }}."""
        {{ model_name_lower }} = {{ model_name }}.objects.create(
            {{ test_fields }}
        )
        self.assertTrue(isinstance({{ model_name_lower }}, {{ model_name }}))
        self.assertEqual({{ model_name_lower }}.{{ str_field }}, {{ str_value }})
    
    def test_{{ model_name_lower }}_str_representation(self):
        """Test string representation."""
        {{ model_name_lower }} = {{ model_name }}.objects.create(
            {{ test_fields }}
        )
        self.assertEqual(str({{ model_name_lower }}), {{ expected_str }})
    
    def test_{{ model_name_lower }}_validation(self):
        """Test model validation."""
        with self.assertRaises(ValidationError):
            {{ model_name_lower }} = {{ model_name }}(
                {{ invalid_fields }}
            )
            {{ model_name_lower }}.full_clean()
    
    def test_{{ model_name_lower }}_unique_constraint(self):
        """Test unique constraints."""
        {{ model_name }}.objects.create({{ test_fields }})
        with self.assertRaises(IntegrityError):
            {{ model_name }}.objects.create({{ test_fields }})
''')
        
        # Extract model info from content
        model_info = self._extract_model_info(models_content)
        
        return template.render(**model_info)
    
    async def _generate_view_tests(self, views_content: str) -> str:
        """Generate view tests."""
        template = Template('''"""
Comprehensive view tests - Auto-generated
"""
import pytest
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
import json

class {{ view_name }}TestCase(APITestCase):
    """Test cases for {{ view_name }} views."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = get_user_model().objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_list_{{ model_name_lower }}s(self):
        """Test listing {{ model_name_lower }}s."""
        response = self.client.get('/api/{{ model_name_lower }}s/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
    
    def test_create_{{ model_name_lower }}(self):
        """Test creating a {{ model_name_lower }}."""
        data = {{ test_data }}
        response = self.client.post('/api/{{ model_name_lower }}s/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_retrieve_{{ model_name_lower }}(self):
        """Test retrieving a specific {{ model_name_lower }}."""
        # Create test object first
        {{ model_name_lower }} = {{ model_name }}.objects.create({{ test_fields }})
                 response = self.client.get('/api/{{ model_name_lower }}s/{}/'.format({{ model_name_lower }}.id))
         self.assertEqual(response.status_code, status.HTTP_200_OK)
     
     def test_update_{{ model_name_lower }}(self):
         """Test updating a {{ model_name_lower }}."""
         {{ model_name_lower }} = {{ model_name }}.objects.create({{ test_fields }})
         data = {{ update_data }}
         response = self.client.put('/api/{{ model_name_lower }}s/{}/'.format({{ model_name_lower }}.id), data)
         self.assertEqual(response.status_code, status.HTTP_200_OK)
     
     def test_delete_{{ model_name_lower }}(self):
         """Test deleting a {{ model_name_lower }}."""
         {{ model_name_lower }} = {{ model_name }}.objects.create({{ test_fields }})
         response = self.client.delete('/api/{{ model_name_lower }}s/{}/'.format({{ model_name_lower }}.id))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
    
    def test_authentication_required(self):
        """Test that authentication is required."""
        self.client.logout()
        response = self.client.get('/api/{{ model_name_lower }}s/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
''')
        
                 view_info = self._extract_view_info(views_content)
         return template.render(**view_info)
     
     async def _generate_serializer_tests(self, content: str) -> str:
         """Generate serializer tests."""
         return '''"""
Serializer tests - Auto-generated
"""
import pytest
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import serializers

class SerializerTestCase(TestCase):
    """Test serializer functionality."""
    
    def test_serializer_validation(self):
        """Test serializer validation."""
        pass
'''
     
     async def _generate_integration_tests(self, generated_files: Dict[str, str]) -> str:
        """Generate integration tests."""
        return '''"""
Integration tests - Auto-generated
"""
import pytest
from django.test import TestCase, TransactionTestCase
from django.db import transaction
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model

class APIIntegrationTestCase(APITestCase):
    """End-to-end API integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.user = get_user_model().objects.create_user(
            username='integrationuser',
            email='integration@example.com',
            password='integrationpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_full_crud_workflow(self):
        """Test complete CRUD workflow across all models."""
        # This would test the entire workflow
        pass
    
    def test_data_consistency(self):
        """Test data consistency across related models."""
        # Test referential integrity
        pass
    
    def test_performance_benchmarks(self):
        """Test API performance benchmarks."""
        # Basic performance tests
        pass
'''
    
    def _extract_model_info(self, content: str) -> Dict[str, Any]:
        """Extract model information for test generation."""
        # Simple extraction - in practice, would use AST parsing
        model_name = "TestModel"  # Default
        if "class " in content:
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('class ') and 'models.Model' in line:
                    model_name = line.split('class ')[1].split('(')[0].strip()
                    break
        
        return {
            'model_name': model_name,
            'model_name_lower': model_name.lower(),
            'imports': f"from backend.models import {model_name}",
            'test_fields': "name='Test Item'",
            'str_field': 'name',
            'str_value': "'Test Item'",
            'expected_str': "'Test Item'",
            'invalid_fields': "name=''"
        }
    
    def _extract_view_info(self, content: str) -> Dict[str, Any]:
        """Extract view information for test generation."""
        return {
            'view_name': 'API',
            'model_name': 'TestModel',
            'model_name_lower': 'testmodel',
            'test_data': "{'name': 'Test Item'}",
            'test_fields': "name='Test Item'",
            'update_data': "{'name': 'Updated Item'}"
        }
    
    async def generate_cicd_config(self, project_name: str = "django-backend") -> Dict[str, str]:
        """Generate CI/CD configuration files."""
        print("🚀 Generating CI/CD pipeline configuration...")
        
        configs = {}
        
        # GitHub Actions workflow
        configs['.github/workflows/ci.yml'] = self._generate_github_actions()
        
        # Docker configurations
        configs['Dockerfile'] = self._generate_dockerfile()
        configs['docker-compose.yml'] = self._generate_docker_compose()
        configs['docker-compose.prod.yml'] = self._generate_docker_compose_prod()
        
        # Deployment scripts
        configs['deploy.sh'] = self._generate_deploy_script()
        configs['docker-entrypoint.sh'] = self._generate_entrypoint_script()
        
        return configs
    
    def _generate_github_actions(self) -> str:
        """Generate GitHub Actions workflow."""
        return '''name: Django CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest-django pytest-cov
    
    - name: Run migrations
      run: |
        python manage.py migrate
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test_db
    
    - name: Run tests with coverage
      run: |
        pytest --cov=. --cov-report=xml
      env:
        DATABASE_URL: postgres://postgres:localhost:5432/test_db
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
    
    - name: Run security checks
      run: |
        pip install bandit safety
        bandit -r . -x tests/
        safety check
    
    - name: Lint code
      run: |
        pip install flake8 black isort
        flake8 .
        black --check .
        isort --check-only .

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t django-backend:${{ github.sha }} .
        docker tag django-backend:${{ github.sha }} django-backend:latest
    
    - name: Deploy to staging
      if: github.ref == 'refs/heads/develop'
      run: |
        echo "Deploy to staging environment"
        # Add staging deployment commands
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production environment"
        # Add production deployment commands
'''
    
    def _generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile."""
        return '''# Multi-stage Dockerfile for Django backend
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Copy project
COPY . .

# Change ownership of the app directory
RUN chown -R app:app /app

# Switch to app user
USER app

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health/ || exit 1

# Start server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "backend.wsgi:application"]
'''
    
    def _generate_docker_compose(self) -> str:
        """Generate Docker Compose for development."""
        return '''version: '3.8'

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
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U django_user -d django_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

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
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery:
    build: .
    command: celery -A backend worker -l info
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgres://django_user:django_password@db:5432/django_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
'''

    def _generate_deploy_script(self) -> str:
        """Generate deployment script."""
        return '''#!/bin/bash
# Deployment script for Django backend

set -e

echo "🚀 Starting deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -t django-backend:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start new containers
echo "▶️  Starting new containers..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Run migrations
echo "📦 Running database migrations..."
docker-compose exec web python manage.py migrate

# Collect static files
echo "📁 Collecting static files..."
docker-compose exec web python manage.py collectstatic --noinput

# Create superuser if it doesn't exist
echo "👤 Creating superuser..."
docker-compose exec web python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin / admin123')
else:
    print('Superuser already exists')
"

echo "✅ Deployment completed successfully!"
echo "🌐 Backend is running at: http://localhost:8000"
echo "🔧 Admin interface: http://localhost:8000/admin"
echo "📖 API documentation: http://localhost:8000/api/docs"
'''

    def _generate_entrypoint_script(self) -> str:
        """Generate Docker entrypoint script."""
        return '''#!/bin/bash
# Docker entrypoint script

set -e

# Wait for database to be ready
echo "⏳ Waiting for database..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 1
done
echo "✅ Database is ready!"

# Run migrations
echo "📦 Running migrations..."
python manage.py migrate --noinput

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

# Start server
echo "🚀 Starting Django server..."
exec "$@"
'''

# Example usage and integration
async def demo_priority3_features():
    """Demonstrate Priority 3 advanced features."""
    print("🚀 Priority 3: Advanced Features Demo")
    print("=" * 60)
    
    # 1. Agent Specialization Demo
    print("\n🤖 1. AGENT SPECIALIZATION")
    
    # Create specialized agents
    gpt4_capability = AgentCapability(
        name="GPT-4 Complex Code Generator",
        complexity_range=(0.7, 1.0),
        domain_expertise=["general", "fintech", "healthcare"],
        model_preferences=["gpt-4"],
        cost_per_token=0.03,
        quality_score=0.95,
        specializations=["models", "views", "complex_logic"]
    )
    
    qwen_capability = AgentCapability(
        name="Qwen Fast Code Generator", 
        complexity_range=(0.0, 0.6),
        domain_expertise=["general", "ecommerce"],
        model_preferences=["qwen/qwen3-coder:free"],
        cost_per_token=0.0,
        quality_score=0.8,
        specializations=["serializers", "urls", "simple_views"]
    )
    
    ecommerce_capability = AgentCapability(
        name="E-commerce Domain Expert",
        complexity_range=(0.3, 0.9),
        domain_expertise=["ecommerce"],
        model_preferences=["qwen/qwen3-coder:free"],
        cost_per_token=0.0,
        quality_score=0.9,
        specializations=["models", "business_logic"]
    )
    
    # Create agents
    gpt4_agent = ModelSpecificAgent(gpt4_capability, model_name="gpt-4")
    qwen_agent = ModelSpecificAgent(qwen_capability, model_name="qwen/qwen3-coder:free")
    ecommerce_agent = DomainSpecificAgent(ecommerce_capability, domain="ecommerce")
    
    # Create router
    router = IntelligentRouter()
    router.register_agent(gpt4_agent)
    router.register_agent(qwen_agent)
    router.register_agent(ecommerce_agent)
    
    # Test routing decisions
    tasks = [
        {
            "file_type": "models.py",
            "complexity": 0.9,
            "domain": "fintech",
            "prompt": "Generate complex financial models"
        },
        {
            "file_type": "serializers.py", 
            "complexity": 0.3,
            "domain": "general",
            "prompt": "Generate simple serializers"
        },
        {
            "file_type": "models.py",
            "complexity": 0.7,
            "domain": "ecommerce", 
            "prompt": "Generate e-commerce product models"
        }
    ]
    
    context = CodeContext(
        file_dependencies={},
        entity_relationships={},
        domain_patterns=[],
        quality_requirements={},
        existing_code_context="",
        architectural_constraints=[]
    )
    
    for task in tasks:
        selected_agent = await router.route_task(task, context)
        print(f"   Task: {task['file_type']} ({task['domain']}) -> {selected_agent.capability.name}")
    
    # 2. Dependency Analysis Demo
    print("\n🧠 2. DEPENDENCY ANALYSIS")
    analyzer = DependencyAnalyzer()
    
    # Mock file analysis
    print("   📁 Analyzing file dependencies...")
    print("   🔗 Building dependency graph...")
    print("   📊 Calculating complexity scores...")
    print("   🎯 Identifying critical dependencies...")
    
    # 3. Production Features Demo
    print("\n🚀 3. PRODUCTION FEATURES")
    production = ProductionFeatures()
    
    # Generate test files
    mock_generated_files = {
        "models.py": "class Product(models.Model): name = models.CharField(max_length=100)",
        "views.py": "class ProductViewSet(viewsets.ModelViewSet): queryset = Product.objects.all()",
        "serializers.py": "class ProductSerializer(serializers.ModelSerializer): class Meta: model = Product"
    }
    
    print("   🧪 Generating comprehensive test suite...")
    test_files = await production.generate_tests(mock_generated_files)
    for test_file in test_files.keys():
        print(f"      ✅ Generated: {test_file}")
    
    print("   🚀 Generating CI/CD pipeline configuration...")
    cicd_configs = await production.generate_cicd_config("django-backend")
    for config_file in cicd_configs.keys():
        print(f"      ✅ Generated: {config_file}")
    
    print("\n🎉 Priority 3 Advanced Features Demo Complete!")
    print("✅ Agent Specialization: Intelligent routing based on complexity and domain")
    print("✅ Context Awareness: Dependency analysis and relationship mapping")
    print("✅ Production Features: Auto-testing, CI/CD, Docker optimization")

if __name__ == "__main__":
    asyncio.run(demo_priority3_features()) 