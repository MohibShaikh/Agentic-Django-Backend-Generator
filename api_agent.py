"""
APIAgent - Domain Expert for REST & GraphQL API Generation
==========================================================

Third specialized agent in Phase 2: Handles API endpoint generation,
REST/GraphQL schemas, serializers, pagination, and API documentation.

Domain Expertise:
- REST API endpoint design
- GraphQL schema generation
- DRF ViewSets and Serializers
- API versioning strategies
- Pagination and filtering
- Rate limiting and throttling
- API documentation (OpenAPI/Swagger)
- Error handling patterns
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from phase2_specialized_agents import (
    BaseDomainAgent, DomainType, ExpertiseLevel, DomainResponse, DomainRequest
)

class APIAgent(BaseDomainAgent):
    """Expert agent for Django REST/GraphQL API generation"""
    
    def __init__(self, expertise_level: ExpertiseLevel = ExpertiseLevel.ARCHITECT):
        super().__init__(DomainType.API_DESIGN, expertise_level)
        
        # Domain-specific capabilities
        self.capabilities = {
            "rest_api_design",
            "graphql_schema_generation",
            "drf_viewsets",
            "custom_serializers",
            "api_versioning",
            "pagination_systems",
            "filtering_backends",
            "rate_limiting",
            "api_documentation",
            "error_handling"
        }
        
        # API design patterns we can implement
        self.api_patterns = {
            "rest_crud": "Standard REST CRUD operations",
            "graphql_schema": "GraphQL schema with resolvers",
            "api_versioning": "URL/Header-based API versioning",
            "nested_resources": "Nested REST resource endpoints",
            "bulk_operations": "Bulk create/update/delete endpoints",
            "filtering": "Advanced filtering and search",
            "pagination": "Cursor/Page-based pagination",
            "rate_limiting": "API rate limiting and throttling"
        }
        
        # Common API requirements
        self.api_formats = {
            "REST": "Django REST Framework ViewSets",
            "GraphQL": "GraphQL schema with Graphene",
            "REST+GraphQL": "Hybrid REST and GraphQL endpoints"
        }
        
        self.client = None
        
    def set_openai_client(self, client: Any):
        """Set OpenAI client for AI-powered generation"""
        self.client = client
        
    async def generate_domain_code(self, requirements: Dict[str, Any]) -> str:
        """Generate API code for Django"""
        print(f"🌐 APIAgent generating code...")
        
        # Extract API requirements
        shared_knowledge = requirements.get("shared_knowledge", {})
        if hasattr(shared_knowledge, 'erd'):
            erd = shared_knowledge.erd
        else:
            erd = shared_knowledge.get("erd", {}) if isinstance(shared_knowledge, dict) else requirements.get("erd", {})
            
        api_requirements = requirements.get("api_requirements", {})
        domain_insights = requirements.get("domain_insights", {})
        
        # Analyze API needs
        analysis = await self._analyze_api_requirements(erd, api_requirements, domain_insights)
        
        # Generate API code
        if self.client:
            code = await self._generate_with_ai(erd, api_requirements, analysis, domain_insights)
        else:
            code = await self._generate_without_ai(erd, api_requirements, analysis)
            
        return code
        
    async def _analyze_api_requirements(self, erd: Dict, api_reqs: Dict, insights: Dict) -> Dict[str, Any]:
        """Analyze API requirements and ERD structure"""
        analysis = {
            "api_format": "REST",
            "entities_for_api": [],
            "custom_endpoints": [],
            "pagination_needed": False,
            "filtering_needed": False,
            "versioning_strategy": "none",
            "bulk_operations": [],
            "nested_resources": [],
            "rate_limiting_needed": False
        }
        
        # Determine API format
        api_format = api_reqs.get("format", "REST").upper()
        analysis["api_format"] = api_format
        
        # Analyze entities for API endpoints
        entities = erd.get("entities", {})
        for entity_name, entity_data in entities.items():
            analysis["entities_for_api"].append({
                "name": entity_name,
                "fields": list(entity_data.get("fields", {}).keys()),
                "relationships": list(entity_data.get("relationships", {}).keys()),
                "needs_crud": True
            })
            
        # Check for pagination requirement
        if api_reqs.get("pagination", False):
            analysis["pagination_needed"] = True
            
        # Check for filtering requirement
        if api_reqs.get("filtering", False):
            analysis["filtering_needed"] = True
            
        # Check versioning
        if api_reqs.get("versioning"):
            analysis["versioning_strategy"] = api_reqs["versioning"]
            
        # Analyze business logic insights for custom endpoints
        business_insights = insights.get("business_logic", {})
        if business_insights and business_insights.get("business_rules_implemented"):
            analysis["custom_endpoints"].append("bulk_operations")
            
        # Check authentication insights for rate limiting
        auth_insights = insights.get("authentication", {})
        if auth_insights and auth_insights.get("auth_ready"):
            analysis["rate_limiting_needed"] = True
            
        return analysis
        
    async def _generate_with_ai(self, erd: Dict, api_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Generate API code using AI"""
        
        prompt = self._create_api_prompt(erd, api_reqs, analysis, insights)
        
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
            return await self._generate_without_ai(erd, api_reqs, analysis)
            
    def _create_api_prompt(self, erd: Dict, api_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Create concise prompt for API generation"""
        
        entities = [e["name"] for e in analysis["entities_for_api"][:2]]
        api_format = analysis["api_format"]
        features = []
        if analysis["pagination_needed"]:
            features.append("pagination")
        if analysis["filtering_needed"]:
            features.append("filtering")
            
        return f"""Generate Django {api_format} API for {entities}. Include {features}. Return Python ViewSets and URLs."""
        
    async def _generate_without_ai(self, erd: Dict, api_reqs: Dict, analysis: Dict) -> str:
        """Generate basic API code without AI"""
        
        code_parts = []
        code_parts.append("# Django REST API Generation")
        code_parts.append("# Generated by APIAgent")
        code_parts.append("")
        
        # Imports
        imports = [
            "from rest_framework import viewsets, permissions, status",
            "from rest_framework.decorators import action",
            "from rest_framework.response import Response",
            "from rest_framework.pagination import PageNumberPagination",
            "from django_filters.rest_framework import DjangoFilterBackend",
            "from rest_framework.filters import SearchFilter, OrderingFilter",
            "from django.shortcuts import get_object_or_404",
            "from django.urls import path, include",
            "from rest_framework.routers import DefaultRouter"
        ]
        
        if analysis["api_format"] == "GRAPHQL":
            imports.extend([
                "import graphene",
                "from graphene_django import DjangoObjectType"
            ])
            
        code_parts.extend(imports)
        code_parts.append("")
        
        # Pagination class if needed
        if analysis["pagination_needed"]:
            code_parts.extend([
                "# Custom Pagination",
                "class StandardResultsSetPagination(PageNumberPagination):",
                "    page_size = 20",
                "    page_size_query_param = 'page_size'",
                "    max_page_size = 100",
                ""
            ])
            
        # Generate ViewSets for REST API
        if analysis["api_format"] in ["REST", "REST+GRAPHQL"]:
            code_parts.append("# REST API ViewSets")
            
            for entity_info in analysis["entities_for_api"]:
                entity_name = entity_info["name"]
                
                code_parts.extend([
                    f"class {entity_name}ViewSet(viewsets.ModelViewSet):",
                    f'    """ViewSet for {entity_name} model with full CRUD operations"""',
                    f"    queryset = {entity_name}.objects.all()",
                    f"    serializer_class = {entity_name}Serializer",
                    "    permission_classes = [permissions.IsAuthenticated]"
                ])
                
                # Add pagination
                if analysis["pagination_needed"]:
                    code_parts.append("    pagination_class = StandardResultsSetPagination")
                    
                # Add filtering
                if analysis["filtering_needed"]:
                    code_parts.extend([
                        "    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]",
                        f"    filterset_fields = {entity_info['fields'][:3]}",
                        f"    search_fields = {entity_info['fields'][:2]}",
                        f"    ordering_fields = {entity_info['fields'][:2]}"
                    ])
                    
                # Add custom actions
                code_parts.extend([
                    "",
                    "    @action(detail=False, methods=['get'])",
                    "    def stats(self, request):",
                    f'        """Get {entity_name} statistics"""',
                    f"        total = {entity_name}.objects.count()",
                    "        return Response({'total': total})",
                    "",
                    "    @action(detail=True, methods=['post'])",
                    "    def custom_action(self, request, pk=None):",
                    f'        """Custom action for {entity_name}"""',
                    f"        {entity_name.lower()} = self.get_object()",
                    "        # Add custom logic here",
                    f"        return Response({{'message': 'Action completed for {entity_name.lower()}'}})",
                    ""
                ])
                
        # Generate GraphQL schema if needed
        if analysis["api_format"] in ["GRAPHQL", "REST+GRAPHQL"]:
            code_parts.append("# GraphQL Schema")
            
            for entity_info in analysis["entities_for_api"]:
                entity_name = entity_info["name"]
                
                code_parts.extend([
                    f"class {entity_name}Type(DjangoObjectType):",
                    "    class Meta:",
                    f"        model = {entity_name}",
                    f"        fields = {entity_info['fields'][:5]}",
                    ""
                ])
                
            # GraphQL Query class
            code_parts.extend([
                "class Query(graphene.ObjectType):",
                "    # GraphQL queries"
            ])
            
            for entity_info in analysis["entities_for_api"]:
                entity_name = entity_info["name"]
                code_parts.extend([
                    f"    all_{entity_name.lower()}s = graphene.List({entity_name}Type)",
                    f"    {entity_name.lower()} = graphene.Field({entity_name}Type, id=graphene.Int())",
                ])
                
            code_parts.extend([
                "",
                "    # Resolvers"
            ])
            
            for entity_info in analysis["entities_for_api"]:
                entity_name = entity_info["name"]
                code_parts.extend([
                    f"    def resolve_all_{entity_name.lower()}s(self, info):",
                    f"        return {entity_name}.objects.all()",
                    "",
                    f"    def resolve_{entity_name.lower()}(self, info, id):",
                    f"        return {entity_name}.objects.get(id=id)",
                    ""
                ])
                
            code_parts.extend([
                "schema = graphene.Schema(query=Query)",
                ""
            ])
            
        # URL configuration
        code_parts.extend([
            "# URL Configuration",
            "router = DefaultRouter()",
        ])
        
        for entity_info in analysis["entities_for_api"]:
            entity_name = entity_info["name"]
            code_parts.append(f"router.register(r'{entity_name.lower()}s', {entity_name}ViewSet)")
            
        code_parts.extend([
            "",
            "# API URL patterns",
            "api_urlpatterns = [",
            "    path('v1/', include(router.urls)),"
        ])
        
        if analysis["api_format"] in ["GRAPHQL", "REST+GRAPHQL"]:
            code_parts.append("    path('graphql/', GraphQLView.as_view(graphiql=True)),")
            
        code_parts.extend([
            "]",
            "",
            "# Usage Instructions:",
            "# 1. Include api_urlpatterns in your main urls.py",
            "# 2. Add DRF settings to settings.py",
            "# 3. Run migrations if new models",
            "# 4. Access API at /api/v1/ for REST",
            "# 5. Access GraphQL at /api/graphql/ if enabled"
        ])
        
        return "\n".join(code_parts)
        
    async def validate_cross_domain(self, other_domain: DomainType, code: str) -> DomainResponse:
        """Validate how other domain code affects API design"""
        
        concerns = []
        suggestions = []
        
        if other_domain == DomainType.AUTHENTICATION:
            # Check if API uses proper authentication
            if "permission_classes" not in code:
                concerns.append("API endpoints should specify permission classes")
                suggestions.append("Add permission_classes to all ViewSets")
                
            if "IsAuthenticated" not in code:
                suggestions.append("Consider adding IsAuthenticated permission")
                
        elif other_domain == DomainType.BUSINESS_LOGIC:
            # Check if API respects business logic
            if "clean(" not in code and "save(" in code:
                concerns.append("API should trigger business logic validation")
                suggestions.append("Call model.clean() before saving in serializers")
                
            if "is_owned_by" in code:
                suggestions.append("Integrate ownership validation with API permissions")
                
        return DomainResponse(
            request_id="validation",
            from_domain=self.domain,
            response_type="approved" if not concerns else "suggestion",
            content={
                "concerns": concerns,
                "suggestions": suggestions,
                "api_quality_score": 0.9 if not concerns else 0.7,
                "performance_considerations": [
                    "Add database indexes for filtered fields",
                    "Consider query optimization for list endpoints",
                    "Add caching for frequently accessed data"
                ]
            },
            confidence=0.85,
            reasoning=f"API design validation for {other_domain.value}"
        )
        
    def provide_domain_insights(self) -> Dict[str, Any]:
        """Provide API design insights for other agents"""
        return {
            "api_patterns": list(self.api_patterns.keys()),
            "supported_formats": ["REST", "GraphQL", "REST+GraphQL"],
            "integration_points": {
                "authentication": "Needs permission classes on all endpoints",
                "business_logic": "Needs business rule enforcement in serializers",
                "testing": "Needs API endpoint test coverage"
            },
            "performance_features": [
                "Pagination",
                "Filtering",
                "Search",
                "Ordering",
                "Bulk operations"
            ],
            "api_ready": True,
            "documentation_generated": True
        }
        
    async def consult_for_api_strategy(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized consultation for API architecture strategy"""
        print(f"🌐 APIAgent analyzing API strategy...")
        
        strategy = {}
        
        # Determine API architecture
        if requirements.get("mobile_app", False) or requirements.get("spa_frontend", False):
            strategy["architecture"] = "API-First"
            strategy["recommendation"] = "Pure REST API with comprehensive endpoints"
        elif requirements.get("complex_queries", False):
            strategy["architecture"] = "GraphQL-First"
            strategy["recommendation"] = "GraphQL with REST fallback for simple operations"
        else:
            strategy["architecture"] = "Hybrid"
            strategy["recommendation"] = "REST for CRUD, GraphQL for complex queries"
            
        # Performance considerations
        entity_count = len(requirements.get("entities", {}))
        if entity_count > 10:
            strategy["performance_priority"] = "high"
            strategy["optimizations"] = ["Database indexing", "Query optimization", "Caching"]
        else:
            strategy["performance_priority"] = "standard"
            strategy["optimizations"] = ["Basic pagination", "Simple filtering"]
            
        # Security level
        if requirements.get("public_api", False):
            strategy["security_level"] = "high"
            strategy["security_features"] = ["Rate limiting", "API keys", "Request validation"]
        else:
            strategy["security_level"] = "standard"
            strategy["security_features"] = ["Authentication", "Permission checks"]
            
        return {
            "strategy": strategy,
            "estimated_endpoints": entity_count * 5,  # CRUD + custom per entity
            "implementation_priority": "Medium - After business logic and auth",
            "dependencies": ["Business logic validation", "Authentication system"]
        }

# ============================================================================
# Integration Helper
# ============================================================================

def create_api_agent(openai_client: Optional[Any] = None) -> APIAgent:
    """Factory function to create configured APIAgent"""
    agent = APIAgent(ExpertiseLevel.ARCHITECT)
    
    if openai_client:
        agent.set_openai_client(openai_client)
        
    print(f"✅ APIAgent created with {len(agent.capabilities)} capabilities")
    return agent

if __name__ == "__main__":
    # Demo usage
    agent = create_api_agent()
    print(f"🌐 APIAgent ready")
    print(f"📋 Capabilities: {', '.join(agent.capabilities)}")
    print(f"🎯 Patterns: {', '.join(agent.api_patterns.keys())}") 