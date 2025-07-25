"""
TestingAgent - Domain Expert for Automated Test Generation
=========================================================

Fourth specialized agent in Phase 2: Handles comprehensive test generation,
unit tests, integration tests, API tests, and testing strategy.

Domain Expertise:
- Django unit testing
- Model and view test generation
- API endpoint testing
- Integration test strategies
- Test data factories
- Performance testing
- Test coverage analysis
- Testing best practices
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from phase2_specialized_agents import (
    BaseDomainAgent, DomainType, ExpertiseLevel, DomainResponse, DomainRequest
)

class TestingAgent(BaseDomainAgent):
    """Expert agent for Django test generation and testing strategy"""
    
    def __init__(self, expertise_level: ExpertiseLevel = ExpertiseLevel.ARCHITECT):
        super().__init__(DomainType.TESTING, expertise_level)
        
        # Domain-specific capabilities
        self.capabilities = {
            "unit_test_generation",
            "integration_testing",
            "api_endpoint_testing",
            "model_testing",
            "view_testing",
            "factory_generation",
            "test_data_creation",
            "performance_testing",
            "coverage_analysis",
            "test_automation"
        }
        
        # Testing patterns we can implement
        self.test_patterns = {
            "model_tests": "Unit tests for Django models",
            "view_tests": "Tests for views and API endpoints",
            "integration_tests": "Cross-component integration tests",
            "factory_pattern": "Test data factories using factory_boy",
            "api_tests": "REST/GraphQL API endpoint tests",
            "performance_tests": "Load and performance testing",
            "security_tests": "Authentication and authorization tests",
            "edge_case_tests": "Boundary and error condition tests"
        }
        
        # Testing frameworks and tools
        self.testing_tools = {
            "django_test": "Django's built-in testing framework",
            "pytest": "Pytest with Django plugin",
            "factory_boy": "Test data factories",
            "coverage": "Code coverage analysis",
            "selenium": "Browser automation testing",
            "locust": "Performance and load testing"
        }
        
        self.client = None
        
    def set_openai_client(self, client: Any):
        """Set OpenAI client for AI-powered generation"""
        self.client = client
        
    async def generate_domain_code(self, requirements: Dict[str, Any]) -> str:
        """Generate comprehensive test suite for Django"""
        print(f"🧪 TestingAgent generating test suite...")
        
        # Extract testing requirements
        shared_knowledge = requirements.get("shared_knowledge", {})
        if hasattr(shared_knowledge, 'erd'):
            erd = shared_knowledge.erd
        else:
            erd = shared_knowledge.get("erd", {}) if isinstance(shared_knowledge, dict) else requirements.get("erd", {})
            
        test_requirements = requirements.get("test_requirements", {})
        domain_insights = requirements.get("domain_insights", {})
        
        # Analyze testing needs
        analysis = await self._analyze_testing_requirements(erd, test_requirements, domain_insights)
        
        # Generate test code
        if self.client:
            code = await self._generate_with_ai(erd, test_requirements, analysis, domain_insights)
        else:
            code = await self._generate_without_ai(erd, test_requirements, analysis)
            
        return code
        
    async def _analyze_testing_requirements(self, erd: Dict, test_reqs: Dict, insights: Dict) -> Dict[str, Any]:
        """Analyze testing requirements and determine test strategy"""
        analysis = {
            "test_types": ["unit", "integration"],
            "models_to_test": [],
            "api_endpoints_to_test": [],
            "business_logic_to_test": [],
            "auth_scenarios_to_test": [],
            "coverage_target": 80,
            "factory_needed": True,
            "performance_tests_needed": False,
            "security_tests_needed": False
        }
        
        # Determine test types based on requirements
        if test_reqs.get("include_api_tests", True):
            analysis["test_types"].append("api")
            
        if test_reqs.get("include_performance_tests", False):
            analysis["test_types"].append("performance")
            analysis["performance_tests_needed"] = True
            
        if test_reqs.get("include_security_tests", False):
            analysis["test_types"].append("security")
            analysis["security_tests_needed"] = True
            
        # Analyze entities for model testing
        entities = erd.get("entities", {})
        for entity_name, entity_data in entities.items():
            analysis["models_to_test"].append({
                "name": entity_name,
                "fields": list(entity_data.get("fields", {}).keys()),
                "relationships": list(entity_data.get("relationships", {}).keys()),
                "has_validation": bool(entity_data.get("validation", {}))
            })
            
        # Analyze API insights for endpoint testing
        api_insights = insights.get("api_design", {})
        if api_insights and api_insights.get("api_ready"):
            for model_info in analysis["models_to_test"]:
                analysis["api_endpoints_to_test"].extend([
                    f"{model_info['name'].lower()}_list",
                    f"{model_info['name'].lower()}_detail",
                    f"{model_info['name'].lower()}_create",
                    f"{model_info['name'].lower()}_update",
                    f"{model_info['name'].lower()}_delete"
                ])
                
        # Analyze business logic insights
        business_insights = insights.get("business_logic", {})
        if business_insights and business_insights.get("business_rules_implemented"):
            analysis["business_logic_to_test"].extend([
                "custom_validation_methods",
                "business_rule_enforcement",
                "data_integrity_checks"
            ])
            
        # Analyze authentication insights
        auth_insights = insights.get("authentication", {})
        if auth_insights and auth_insights.get("auth_ready"):
            analysis["auth_scenarios_to_test"].extend([
                "user_registration",
                "user_login",
                "permission_checking",
                "role_based_access"
            ])
            analysis["security_tests_needed"] = True
            
        # Set coverage target based on requirements
        analysis["coverage_target"] = test_reqs.get("coverage_target", 80)
        
        return analysis
        
    async def _generate_with_ai(self, erd: Dict, test_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Generate test code using AI"""
        
        prompt = self._create_testing_prompt(erd, test_reqs, analysis, insights)
        
        try:
            if self.client and hasattr(self.client, 'chat'):
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=70
                )
                return response.choices[0].message.content.strip()
            else:
                raise Exception("OpenAI client not properly configured")
        except Exception as e:
            print(f"❌ AI generation failed: {e}")
            return await self._generate_without_ai(erd, test_reqs, analysis)
            
    def _create_testing_prompt(self, erd: Dict, test_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Create concise prompt for test generation"""
        
        models = [m["name"] for m in analysis["models_to_test"][:2]]
        test_types = analysis["test_types"][:2]
        
        return f"""Generate Django tests for {models}. Include {test_types} tests. Return Python test classes."""
        
    async def _generate_without_ai(self, erd: Dict, test_reqs: Dict, analysis: Dict) -> str:
        """Generate comprehensive test suite without AI"""
        
        code_parts = []
        code_parts.append("# Comprehensive Django Test Suite")
        code_parts.append("# Generated by TestingAgent")
        code_parts.append("")
        
        # Imports
        imports = [
            "import pytest",
            "from django.test import TestCase, TransactionTestCase, Client",
            "from django.contrib.auth import get_user_model",
            "from django.urls import reverse",
            "from django.core.exceptions import ValidationError",
            "from rest_framework.test import APITestCase, APIClient",
            "from rest_framework import status",
            "from unittest.mock import patch, MagicMock",
            "import factory",
            "from factory.django import DjangoModelFactory",
            "import json"
        ]
        
        # Add model imports
        for model_info in analysis["models_to_test"]:
            imports.append(f"from .models import {model_info['name']}")
            
        code_parts.extend(imports)
        code_parts.append("")
        code_parts.append("User = get_user_model()")
        code_parts.append("")
        
        # Generate test factories if needed
        if analysis["factory_needed"]:
            code_parts.append("# Test Data Factories")
            code_parts.append("# ==================")
            code_parts.append("")
            
            # User factory
            code_parts.extend([
                "class UserFactory(DjangoModelFactory):",
                "    class Meta:",
                "        model = User",
                "",
                "    username = factory.Sequence(lambda n: f'user{n}')",
                "    email = factory.LazyAttribute(lambda obj: f'{obj.username}@example.com')",
                "    first_name = factory.Faker('first_name')",
                "    last_name = factory.Faker('last_name')",
                ""
            ])
            
            # Model factories
            for model_info in analysis["models_to_test"]:
                model_name = model_info["name"]
                code_parts.extend([
                    f"class {model_name}Factory(DjangoModelFactory):",
                    "    class Meta:",
                    f"        model = {model_name}",
                    ""
                ])
                
                # Add factory fields based on model fields
                for field in model_info["fields"][:5]:  # Limit to first 5 fields
                    if field.lower() in ["name", "title"]:
                        code_parts.append(f"    {field} = factory.Faker('name')")
                    elif field.lower() in ["email"]:
                        code_parts.append(f"    {field} = factory.Faker('email')")
                    elif field.lower() in ["description", "content", "text"]:
                        code_parts.append(f"    {field} = factory.Faker('text')")
                    elif field.lower() in ["slug"]:
                        code_parts.append(f"    {field} = factory.Faker('slug')")
                    elif field.lower() in ["user", "author", "owner"]:
                        code_parts.append(f"    {field} = factory.SubFactory(UserFactory)")
                    else:
                        code_parts.append(f"    {field} = factory.Faker('word')")
                        
                code_parts.append("")
                
        # Generate model tests
        if "unit" in analysis["test_types"]:
            code_parts.append("# Model Unit Tests")
            code_parts.append("# ================")
            code_parts.append("")
            
            for model_info in analysis["models_to_test"]:
                model_name = model_info["name"]
                
                code_parts.extend([
                    f"class {model_name}ModelTest(TestCase):",
                    f'    """Test cases for {model_name} model"""',
                    "",
                    "    def setUp(self):",
                    "        self.user = UserFactory()",
                    f"        self.{model_name.lower()} = {model_name}Factory()",
                    "",
                    f"    def test_{model_name.lower()}_creation(self):",
                    f'        """Test {model_name} instance creation"""',
                    f"        self.assertTrue(isinstance(self.{model_name.lower()}, {model_name}))",
                    f"        self.assertEqual({model_name}.objects.count(), 1)",
                    "",
                    f"    def test_{model_name.lower()}_str_representation(self):",
                    f'        """Test {model_name} string representation"""',
                    f"        expected = str(self.{model_name.lower()})",
                    f"        self.assertEqual(str(self.{model_name.lower()}), expected)",
                    ""
                ])
                
                # Add validation tests if model has validation
                if model_info["has_validation"]:
                    code_parts.extend([
                        f"    def test_{model_name.lower()}_validation(self):",
                        f'        """Test {model_name} validation rules"""',
                        f"        # Test invalid data",
                        f"        invalid_{model_name.lower()} = {model_name}()",
                        "        with self.assertRaises(ValidationError):",
                        f"            invalid_{model_name.lower()}.full_clean()",
                        ""
                    ])
                    
                # Add relationship tests
                if model_info["relationships"]:
                    code_parts.extend([
                        f"    def test_{model_name.lower()}_relationships(self):",
                        f'        """Test {model_name} model relationships"""',
                        f"        # Test relationships exist",
                        f"        self.assertTrue(hasattr(self.{model_name.lower()}, '{model_info['relationships'][0]}'))",
                        ""
                    ])
                    
                code_parts.append("")
                
        # Generate API tests
        if "api" in analysis["test_types"]:
            code_parts.append("# API Endpoint Tests")
            code_parts.append("# ==================")
            code_parts.append("")
            
            for model_info in analysis["models_to_test"]:
                model_name = model_info["name"]
                model_lower = model_name.lower()
                
                code_parts.extend([
                    f"class {model_name}APITest(APITestCase):",
                    f'    """Test cases for {model_name} API endpoints"""',
                    "",
                    "    def setUp(self):",
                    "        self.client = APIClient()",
                    "        self.user = UserFactory()",
                    f"        self.{model_lower} = {model_name}Factory()",
                    f"        self.list_url = reverse('{model_lower}-list')",
                    f"        self.detail_url = reverse('{model_lower}-detail', kwargs={{'pk': self.{model_lower}.pk}})",
                    "",
                    f"    def test_{model_lower}_list_authenticated(self):",
                    f'        """Test {model_name} list endpoint with authentication"""',
                    "        self.client.force_authenticate(user=self.user)",
                    "        response = self.client.get(self.list_url)",
                    "        self.assertEqual(response.status_code, status.HTTP_200_OK)",
                    f"        self.assertEqual(len(response.data['results']), 1)",
                    "",
                    f"    def test_{model_lower}_list_unauthenticated(self):",
                    f'        """Test {model_name} list endpoint without authentication"""',
                    "        response = self.client.get(self.list_url)",
                    "        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)",
                    "",
                    f"    def test_{model_lower}_detail_retrieve(self):",
                    f'        """Test {model_name} detail retrieval"""',
                    "        self.client.force_authenticate(user=self.user)",
                    "        response = self.client.get(self.detail_url)",
                    "        self.assertEqual(response.status_code, status.HTTP_200_OK)",
                    f"        self.assertEqual(response.data['id'], self.{model_lower}.id)",
                    "",
                    f"    def test_{model_lower}_create(self):",
                    f'        """Test {model_name} creation via API"""',
                    "        self.client.force_authenticate(user=self.user)",
                    f"        data = {model_name}Factory.build().__dict__",
                    "        data.pop('id', None)  # Remove id if present",
                    "        response = self.client.post(self.list_url, data, format='json')",
                    "        self.assertEqual(response.status_code, status.HTTP_201_CREATED)",
                    f"        self.assertEqual({model_name}.objects.count(), 2)",
                    "",
                    f"    def test_{model_lower}_update(self):",
                    f'        """Test {model_name} update via API"""',
                    "        self.client.force_authenticate(user=self.user)",
                    "        data = {'name': 'Updated Name'}  # Adjust field as needed",
                    "        response = self.client.patch(self.detail_url, data, format='json')",
                    "        self.assertEqual(response.status_code, status.HTTP_200_OK)",
                    "",
                    f"    def test_{model_lower}_delete(self):",
                    f'        """Test {model_name} deletion via API"""',
                    "        self.client.force_authenticate(user=self.user)",
                    "        response = self.client.delete(self.detail_url)",
                    "        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)",
                    f"        self.assertEqual({model_name}.objects.count(), 0)",
                    ""
                ])
                
        # Generate business logic tests
        if analysis["business_logic_to_test"]:
            code_parts.append("# Business Logic Tests")
            code_parts.append("# ====================")
            code_parts.append("")
            
            code_parts.extend([
                "class BusinessLogicTest(TestCase):",
                '    """Test cases for business logic and validation"""',
                "",
                "    def setUp(self):",
                "        self.user = UserFactory()",
                "",
                "    def test_custom_validation_methods(self):",
                '        """Test custom model validation methods"""',
                "        # Add specific business logic tests here",
                "        pass",
                "",
                "    def test_business_rule_enforcement(self):",
                '        """Test business rule enforcement"""',
                "        # Add business rule tests here",
                "        pass",
                "",
                "    def test_data_integrity_checks(self):",
                '        """Test data integrity constraints"""',
                "        # Add data integrity tests here",
                "        pass",
                ""
            ])
            
        # Generate authentication tests
        if analysis["auth_scenarios_to_test"]:
            code_parts.append("# Authentication & Authorization Tests")
            code_parts.append("# ===================================")
            code_parts.append("")
            
            code_parts.extend([
                "class AuthenticationTest(TestCase):",
                '    """Test cases for authentication and authorization"""',
                "",
                "    def setUp(self):",
                "        self.client = Client()",
                "        self.user = UserFactory()",
                "",
                "    def test_user_registration(self):",
                '        """Test user registration process"""',
                "        data = {",
                "            'username': 'testuser',",
                "            'email': 'test@example.com',",
                "            'password': 'testpass123'",
                "        }",
                "        # Add registration test logic",
                "        pass",
                "",
                "    def test_user_login(self):",
                '        """Test user login process"""',
                "        login_successful = self.client.login(",
                "            username=self.user.username,",
                "            password='testpass123'",
                "        )",
                "        self.assertTrue(login_successful)",
                "",
                "    def test_permission_checking(self):",
                '        """Test permission-based access control"""',
                "        # Add permission tests here",
                "        pass",
                "",
                "    def test_role_based_access(self):",
                '        """Test role-based access control"""',
                "        # Add RBAC tests here",
                "        pass",
                ""
            ])
            
        # Generate performance tests if needed
        if analysis["performance_tests_needed"]:
            code_parts.append("# Performance Tests")
            code_parts.append("# =================")
            code_parts.append("")
            
            code_parts.extend([
                "class PerformanceTest(TransactionTestCase):",
                '    """Performance and load testing"""',
                "",
                "    def test_bulk_operations_performance(self):",
                '        """Test performance of bulk operations"""',
                "        import time",
                "        start_time = time.time()",
                "",
                "        # Create multiple objects",
                f"        objects = [{analysis['models_to_test'][0]['name']}Factory() for _ in range(100)]",
                "",
                "        end_time = time.time()",
                "        execution_time = end_time - start_time",
                "        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds",
                "",
                "    def test_database_query_optimization(self):",
                '        """Test database query efficiency"""',
                "        from django.test.utils import override_settings",
                "        from django.db import connection",
                "",
                "        with override_settings(DEBUG=True):",
                "            initial_queries = len(connection.queries)",
                "            # Perform operations that should be optimized",
                f"            list({analysis['models_to_test'][0]['name']}.objects.all())",
                "            final_queries = len(connection.queries)",
                "            query_count = final_queries - initial_queries",
                "            self.assertLessEqual(query_count, 5)  # Should use minimal queries",
                ""
            ])
            
        # Add test configuration and utilities
        code_parts.extend([
            "# Test Configuration & Utilities",
            "# ===============================",
            "",
            "class TestUtilities:",
            '    """Utility methods for testing"""',
            "",
            "    @staticmethod",
            "    def create_test_user(username='testuser', email='test@example.com'):",
            '        """Create a test user with default values"""',
            "        return UserFactory(username=username, email=email)",
            "",
            "    @staticmethod",
            "    def authenticate_user(client, user):",
            '        """Authenticate a user for API tests"""',
            "        client.force_authenticate(user=user)",
            "        return client",
            "",
            "    @staticmethod",
            "    def assert_response_structure(response, expected_keys):",
            '        """Assert that response contains expected keys"""',
            "        response_data = response.json() if hasattr(response, 'json') else response.data",
            "        for key in expected_keys:",
            "            assert key in response_data, f'Missing key: {key}'",
            "",
            "",
            "# Test Discovery and Execution",
            "# ============================",
            "",
            "# Run with: python manage.py test",
            "# Run with coverage: coverage run --source='.' manage.py test",
            "# Generate coverage report: coverage report -m",
            "",
            f"# Target Coverage: {analysis['coverage_target']}%",
            f"# Test Types: {', '.join(analysis['test_types'])}",
            f"# Models Tested: {len(analysis['models_to_test'])}",
            f"# API Endpoints Tested: {len(analysis['api_endpoints_to_test'])}"
        ])
        
        return "\n".join(code_parts)
        
    async def validate_cross_domain(self, other_domain: DomainType, code: str) -> DomainResponse:
        """Validate how other domain code affects testing strategy"""
        
        concerns = []
        suggestions = []
        
        if other_domain == DomainType.API_DESIGN:
            # Check if API code needs comprehensive testing
            if "ViewSet" in code and "test" not in code.lower():
                concerns.append("API endpoints need comprehensive test coverage")
                suggestions.append("Add API endpoint tests for all ViewSets")
                
            if "permission_classes" in code:
                suggestions.append("Add authentication and authorization tests")
                
        elif other_domain == DomainType.BUSINESS_LOGIC:
            # Check if business logic needs testing
            if "def clean(" in code or "def save(" in code:
                suggestions.append("Add tests for custom validation and save methods")
                
            if "ValidationError" in code:
                suggestions.append("Add edge case tests for validation errors")
                
        elif other_domain == DomainType.AUTHENTICATION:
            # Check if auth code needs security testing
            if "JWT" in code or "Token" in code:
                suggestions.append("Add token-based authentication tests")
                
            if "permission" in code.lower():
                suggestions.append("Add comprehensive permission testing")
                
        return DomainResponse(
            request_id="validation",
            from_domain=self.domain,
            response_type="approved" if not concerns else "suggestion",
            content={
                "concerns": concerns,
                "suggestions": suggestions,
                "test_coverage_recommendations": [
                    "Aim for >80% code coverage",
                    "Include edge cases and error scenarios",
                    "Test both positive and negative paths",
                    "Add performance tests for critical operations"
                ],
                "testing_quality_score": 0.9 if not concerns else 0.7
            },
            confidence=0.88,
            reasoning=f"Testing strategy validation for {other_domain.value}"
        )
        
    def provide_domain_insights(self) -> Dict[str, Any]:
        """Provide testing insights for other agents"""
        return {
            "test_patterns": list(self.test_patterns.keys()),
            "testing_tools": list(self.testing_tools.keys()),
            "coverage_capabilities": [
                "Unit testing",
                "Integration testing", 
                "API testing",
                "Performance testing",
                "Security testing"
            ],
            "test_automation": {
                "factory_generation": True,
                "test_data_creation": True,
                "coverage_analysis": True,
                "continuous_testing": True
            },
            "integration_points": {
                "models": "Requires comprehensive model testing",
                "api": "Needs API endpoint test coverage",
                "business_logic": "Requires validation and business rule tests",
                "authentication": "Needs security and permission tests"
            },
            "testing_ready": True,
            "comprehensive_coverage": True
        }
        
    async def consult_for_testing_strategy(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized consultation for testing strategy"""
        print(f"🧪 TestingAgent analyzing testing strategy...")
        
        strategy = {}
        
        # Determine testing approach
        entity_count = len(requirements.get("entities", {}))
        has_api = requirements.get("api_requirements", {}).get("format") is not None
        has_complex_logic = bool(requirements.get("business_rules", {}))
        
        if entity_count > 5 or has_api:
            strategy["approach"] = "Comprehensive"
            strategy["priority"] = ["Unit tests", "API tests", "Integration tests"]
        else:
            strategy["approach"] = "Focused"
            strategy["priority"] = ["Unit tests", "Business logic tests"]
            
        # Coverage strategy
        if has_complex_logic:
            strategy["coverage_target"] = 90
            strategy["focus_areas"] = ["Business logic", "Data validation", "Edge cases"]
        else:
            strategy["coverage_target"] = 80
            strategy["focus_areas"] = ["Core functionality", "API endpoints"]
            
        # Testing tools recommendation
        strategy["recommended_tools"] = ["pytest", "factory_boy", "coverage"]
        
        if has_api:
            strategy["recommended_tools"].append("django-rest-framework-testing")
            
        if requirements.get("performance_requirements", False):
            strategy["recommended_tools"].extend(["locust", "django-debug-toolbar"])
            
        return {
            "strategy": strategy,
            "estimated_test_files": max(2, entity_count),
            "estimated_test_cases": entity_count * 8,  # ~8 tests per model on average
            "implementation_priority": "High - Parallel with development",
            "dependencies": ["Models", "Business logic", "API endpoints"]
        }

# ============================================================================
# Integration Helper
# ============================================================================

def create_testing_agent(openai_client: Optional[Any] = None) -> TestingAgent:
    """Factory function to create configured TestingAgent"""
    agent = TestingAgent(ExpertiseLevel.ARCHITECT)
    
    if openai_client:
        agent.set_openai_client(openai_client)
        
    print(f"✅ TestingAgent created with {len(agent.capabilities)} capabilities")
    return agent

if __name__ == "__main__":
    # Demo usage
    agent = create_testing_agent()
    print(f"🧪 TestingAgent ready")
    print(f"📋 Capabilities: {', '.join(agent.capabilities)}")
    print(f"🎯 Patterns: {', '.join(agent.test_patterns.keys())}") 