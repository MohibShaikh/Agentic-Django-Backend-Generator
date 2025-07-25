"""
AuthenticationAgent - Domain Expert for Authentication & Authorization
====================================================================

Second specialized agent in Phase 2: Handles authentication systems,
permission frameworks, JWT tokens, OAuth integration, and Django auth patterns.

Domain Expertise:
- Django authentication backends
- JWT token management
- OAuth2/OpenID Connect integration
- Permission classes and policies
- User roles and groups
- Session management
- API authentication strategies
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from phase2_specialized_agents import (
    BaseDomainAgent, DomainType, ExpertiseLevel, DomainResponse, DomainRequest
)

class AuthenticationAgent(BaseDomainAgent):
    """Expert agent for Django authentication & authorization"""
    
    def __init__(self, expertise_level: ExpertiseLevel = ExpertiseLevel.SPECIALIST):
        super().__init__(DomainType.AUTHENTICATION, expertise_level)
        
        # Domain-specific capabilities
        self.capabilities = {
            "jwt_authentication",
            "oauth2_integration", 
            "permission_classes",
            "custom_auth_backends",
            "role_based_permissions",
            "api_authentication",
            "session_management",
            "user_role_systems"
        }
        
        # Authentication patterns we can implement
        self.auth_patterns = {
            "jwt_api_auth": "JWT tokens for API authentication",
            "oauth2_social": "OAuth2 social login integration",
            "role_permissions": "Role-based permission system",
            "object_permissions": "Object-level permissions",
            "api_key_auth": "API key authentication",
            "session_auth": "Traditional session authentication",
            "multi_factor": "Multi-factor authentication",
            "custom_backends": "Custom authentication backends"
        }
        
        # Common auth requirements
        self.auth_requirements_patterns = {
            "IsAuthenticated": "User must be logged in",
            "IsOwnerOrReadOnly": "Owner can edit, others read-only",
            "IsAdminUser": "Admin users only",
            "IsStaffUser": "Staff users only",
            "HasPermission": "Specific permission required",
            "RoleRequired": "Specific role required"
        }
        
        self.client = None
        
    def set_openai_client(self, client: Any):
        """Set OpenAI client for AI-powered generation"""
        self.client = client
        
    async def generate_domain_code(self, requirements: Dict[str, Any]) -> str:
        """Generate authentication code for Django"""
        print(f"🔐 AuthenticationAgent generating code...")
        
        # Extract auth requirements
        shared_knowledge = requirements.get("shared_knowledge", {})
        if hasattr(shared_knowledge, 'auth_requirements'):
            auth_reqs = shared_knowledge.auth_requirements
        else:
            auth_reqs = shared_knowledge.get("auth_requirements", {}) if isinstance(shared_knowledge, dict) else requirements.get("auth_requirements", {})
            
        domain_insights = requirements.get("domain_insights", {})
        
        # Analyze authentication needs
        analysis = await self._analyze_auth_requirements(auth_reqs, domain_insights)
        
        # Generate authentication code
        if self.client:
            code = await self._generate_with_ai(auth_reqs, analysis, domain_insights)
        else:
            code = await self._generate_without_ai(auth_reqs, analysis)
            
        return code
        
    async def _analyze_auth_requirements(self, auth_reqs: Dict, insights: Dict) -> Dict[str, Any]:
        """Analyze authentication requirements"""
        analysis = {
            "auth_backend_needed": False,
            "jwt_required": False,
            "oauth_required": False,
            "permission_classes": [],
            "custom_permissions": [],
            "role_system": False,
            "api_auth_strategy": "session"
        }
        
        # Check authentication type
        auth_type = auth_reqs.get("authentication", "").lower()
        if "jwt" in auth_type:
            analysis["jwt_required"] = True
            analysis["api_auth_strategy"] = "jwt"
            analysis["auth_backend_needed"] = True
            
        if "oauth" in auth_type:
            analysis["oauth_required"] = True
            analysis["auth_backend_needed"] = True
            
        # Analyze permissions
        permissions = auth_reqs.get("permissions", [])
        for perm in permissions:
            if perm in self.auth_requirements_patterns:
                analysis["permission_classes"].append(perm)
            else:
                analysis["custom_permissions"].append(perm)
                
        # Check for roles
        roles = auth_reqs.get("roles", [])
        if roles:
            analysis["role_system"] = True
            analysis["permission_classes"].append("RoleRequired")
            
        # Check insights from business logic
        business_insights = insights.get("business_logic", {})
        if business_insights and business_insights.get("business_rules_implemented"):
            analysis["custom_permissions"].append("OwnershipValidation")
            
        return analysis
        
    async def _generate_with_ai(self, auth_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Generate authentication code using AI"""
        
        prompt = self._create_auth_prompt(auth_reqs, analysis, insights)
        
        try:
            if self.client and hasattr(self.client, 'chat'):
                response = await self.client.chat.completions.create(
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
            return await self._generate_without_ai(auth_reqs, analysis)
            
    def _create_auth_prompt(self, auth_reqs: Dict, analysis: Dict, insights: Dict) -> str:
        """Create structured prompt for authentication generation"""
        
        auth_type = auth_reqs.get("authentication", "JWT")[:10]
        perms = auth_reqs.get("permissions", [])[:2]
        
        return f"""Generate Django {auth_type} auth with permissions {perms}. Return Python code for JWT views and permission classes."""
        
    async def _generate_without_ai(self, auth_reqs: Dict, analysis: Dict) -> str:
        """Generate basic authentication code without AI"""
        
        code_parts = []
        code_parts.append("# Django Authentication & Authorization")
        code_parts.append("# Generated by AuthenticationAgent")
        code_parts.append("")
        
        # Imports
        imports = [
            "from django.contrib.auth.models import AbstractUser, Group, Permission",
            "from rest_framework.permissions import BasePermission, IsAuthenticated",
            "from rest_framework.authentication import TokenAuthentication",
            "from django.contrib.auth.backends import ModelBackend",
            "from django.db import models"
        ]
        
        if analysis.get("jwt_required"):
            imports.extend([
                "from rest_framework_simplejwt.authentication import JWTAuthentication",
                "from rest_framework_simplejwt.tokens import RefreshToken"
            ])
            
        code_parts.extend(imports)
        code_parts.append("")
        
        # JWT Authentication if needed
        if analysis.get("jwt_required"):
            code_parts.extend([
                "# JWT Authentication Configuration",
                "class JWTAuthenticationMiddleware:",
                '    """JWT Authentication for API endpoints"""',
                "    def __init__(self, get_response):",
                "        self.get_response = get_response",
                "",
                "    def __call__(self, request):",
                "        response = self.get_response(request)",
                "        return response",
                "",
                "# JWT Token Helper",
                "def get_tokens_for_user(user):",
                '    """Generate JWT tokens for user"""',
                "    refresh = RefreshToken.for_user(user)",
                "    return {",
                "        'refresh': str(refresh),",
                "        'access': str(refresh.access_token),",
                "    }",
                ""
            ])
            
        # Permission Classes
        if analysis.get("permission_classes"):
            code_parts.append("# Custom Permission Classes")
            
            for perm_class in analysis["permission_classes"]:
                if perm_class == "IsOwnerOrReadOnly":
                    code_parts.extend([
                        "class IsOwnerOrReadOnly(BasePermission):",
                        '    """Object-level permission to only allow owners to edit"""',
                        "",
                        "    def has_object_permission(self, request, view, obj):",
                        "        # Read permissions for any request",
                        "        if request.method in ['GET', 'HEAD', 'OPTIONS']:",
                        "            return True",
                        "",
                        "        # Write permissions only to owner",
                        "        if hasattr(obj, 'is_owned_by'):",
                        "            return obj.is_owned_by(request.user)",
                        "        return hasattr(obj, 'user') and obj.user == request.user",
                        ""
                    ])
                    
                elif perm_class == "RoleRequired":
                    code_parts.extend([
                        "class RoleRequired(BasePermission):",
                        '    """Permission class for role-based access"""',
                        "",
                        "    def has_permission(self, request, view):",
                        "        if not request.user.is_authenticated:",
                        "            return False",
                        "        return hasattr(request.user, 'role') and request.user.role in getattr(view, 'required_roles', [])",
                        ""
                    ])
                    
        # Role System if needed
        if analysis.get("role_system"):
            roles = auth_reqs.get("roles", ["user", "admin"])
            code_parts.extend([
                "# Role-Based User Model Extension", 
                "class UserRole(models.TextChoices):",
                '    """User role choices"""'
            ])
            
            for role in roles:
                role_upper = role.upper()
                code_parts.append(f'    {role_upper} = "{role}", "{role.title()}"')
                
            code_parts.extend([
                "",
                "class CustomUser(AbstractUser):",
                '    """Extended user model with roles"""',
                "    role = models.CharField(",
                "        max_length=20,",
                "        choices=UserRole.choices,",
                "        default=UserRole.USER",
                "    )",
                "",
                "    def has_role(self, role):",
                '        """Check if user has specific role"""',
                "        return self.role == role",
                ""
            ])
            
        # Authentication Views
        if analysis.get("jwt_required"):
            code_parts.extend([
                "# Authentication Views",
                "from rest_framework.decorators import api_view, permission_classes",
                "from rest_framework.response import Response",
                "from rest_framework import status",
                "from django.contrib.auth import authenticate",
                "",
                "@api_view(['POST'])",
                "@permission_classes([])",
                "def login_view(request):",
                '    """JWT login endpoint"""',
                "    username = request.data.get('username')",
                "    password = request.data.get('password')",
                "",
                "    if not username or not password:",
                "        return Response({'error': 'Username and password required'}, ",
                "                       status=status.HTTP_400_BAD_REQUEST)",
                "",
                "    user = authenticate(username=username, password=password)",
                "    if user:",
                "        tokens = get_tokens_for_user(user)",
                "        return Response(tokens, status=status.HTTP_200_OK)",
                "",
                "    return Response({'error': 'Invalid credentials'}, ",
                "                   status=status.HTTP_401_UNAUTHORIZED)",
                ""
            ])
            
        # Settings recommendations
        code_parts.extend([
            "# Django Settings Configuration",
            "# Add to settings.py:",
            "#",
            "# INSTALLED_APPS = [",
            "#     'rest_framework',",
            "#     'rest_framework_simplejwt'," if analysis.get("jwt_required") else "#     # Add rest_framework if using API auth",
            "#     # ... your apps",
            "# ]",
            "#",
            "# REST_FRAMEWORK = {",
            "#     'DEFAULT_AUTHENTICATION_CLASSES': [",
        ])
        
        if analysis.get("jwt_required"):
            code_parts.append("#         'rest_framework_simplejwt.authentication.JWTAuthentication',")
        else:
            code_parts.append("#         'rest_framework.authentication.SessionAuthentication',")
            
        code_parts.extend([
            "#     ],",
            "#     'DEFAULT_PERMISSION_CLASSES': [",
            "#         'rest_framework.permissions.IsAuthenticated',",
            "#     ],",
            "# }",
            ""
        ])
        
        # Usage instructions
        code_parts.extend([
            "# Usage Instructions:",
            "# 1. Add CustomUser to AUTH_USER_MODEL in settings" if analysis.get("role_system") else "# 1. Configure authentication in settings.py",
            "# 2. Apply migrations for user model changes" if analysis.get("role_system") else "# 2. Use permission classes in your views", 
            "# 3. Use permission classes in DRF views",
            "# 4. Configure JWT settings if using JWT" if analysis.get("jwt_required") else "# 4. Test authentication endpoints",
            "# 5. Integrate with business logic ownership methods"
        ])
        
        return "\n".join(code_parts)
        
    async def validate_cross_domain(self, other_domain: DomainType, code: str) -> DomainResponse:
        """Validate how other domain code affects authentication"""
        
        concerns = []
        suggestions = []
        
        if other_domain == DomainType.BUSINESS_LOGIC:
            # Check if business logic includes ownership validation
            if "is_owned_by" in code.lower() or "ownership" in code.lower():
                suggestions.append("Integrate ownership methods with IsOwnerOrReadOnly permission")
            else:
                concerns.append("Business logic should include ownership validation for permissions")
                
        elif other_domain == DomainType.API_DESIGN:
            # Check if API uses proper authentication
            if "@api_view" in code and "permission_classes" not in code:
                concerns.append("API endpoints should specify permission classes")
                suggestions.append("Add permission_classes decorator to API views")
                
            if "serializer" in code.lower() and "user" not in code.lower():
                suggestions.append("Consider adding user context to serializers")
                
        return DomainResponse(
            request_id="validation",
            from_domain=self.domain,
            response_type="approved" if not concerns else "suggestion",
            content={
                "concerns": concerns,
                "suggestions": suggestions,
                "security_score": 0.9 if not concerns else 0.6,
                "auth_integration_ready": len(concerns) == 0
            },
            confidence=0.85,
            reasoning=f"Authentication security validation for {other_domain.value}"
        )
        
    def provide_domain_insights(self) -> Dict[str, Any]:
        """Provide authentication insights for other agents"""
        return {
            "auth_patterns": list(self.auth_patterns.keys()),
            "permission_framework": [
                "IsAuthenticated",
                "IsOwnerOrReadOnly", 
                "RoleRequired",
                "CustomPermissions"
            ],
            "integration_points": {
                "business_logic": "Needs ownership validation integration",
                "api_design": "Needs permission classes on endpoints",
                "testing": "Needs auth test scenarios with roles"
            },
            "security_features": {
                "jwt_tokens": "API authentication",
                "role_permissions": "Access control",
                "object_permissions": "Resource-level security",
                "auth_backends": "Custom authentication"
            },
            "auth_ready": True
        }
        
    async def consult_for_auth_strategy(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized consultation for authentication strategy"""
        print(f"🔐 AuthenticationAgent analyzing auth strategy...")
        
        strategy = {}
        
        # Determine auth type
        if requirements.get("api_only", False):
            strategy["primary_auth"] = "JWT"
            strategy["session_auth"] = False
        elif requirements.get("web_and_api", True):
            strategy["primary_auth"] = "Hybrid (Session + JWT)"
            strategy["session_auth"] = True
        else:
            strategy["primary_auth"] = "Session"
            strategy["session_auth"] = True
            
        # Role complexity
        roles = requirements.get("roles", [])
        if len(roles) > 3:
            strategy["role_complexity"] = "high"
            strategy["recommendation"] = "Use django-guardian for object permissions"
        elif len(roles) > 0:
            strategy["role_complexity"] = "medium"
            strategy["recommendation"] = "Use Django groups and custom permission classes"
        else:
            strategy["role_complexity"] = "low"
            strategy["recommendation"] = "Use built-in Django permissions"
            
        # Security level
        if requirements.get("sensitive_data", False):
            strategy["security_level"] = "high"
            strategy["additional_features"] = ["Multi-factor auth", "API rate limiting", "Audit logging"]
        else:
            strategy["security_level"] = "standard"
            strategy["additional_features"] = ["Password validation", "Session security"]
            
        return {
            "strategy": strategy,
            "estimated_complexity": strategy["role_complexity"],
            "implementation_priority": "High - Authentication is foundational",
            "dependencies": ["Business logic ownership methods"]
        }

# ============================================================================
# Integration Helper
# ============================================================================

def create_authentication_agent(openai_client: Optional[Any] = None) -> AuthenticationAgent:
    """Factory function to create configured AuthenticationAgent"""
    agent = AuthenticationAgent(ExpertiseLevel.SPECIALIST)
    
    if openai_client:
        agent.set_openai_client(openai_client)
        
    print(f"✅ AuthenticationAgent created with {len(agent.capabilities)} capabilities")
    return agent

if __name__ == "__main__":
    # Demo usage
    agent = create_authentication_agent()
    print(f"🔐 AuthenticationAgent ready")
    print(f"📋 Capabilities: {', '.join(agent.capabilities)}")
    print(f"🎯 Patterns: {', '.join(agent.auth_patterns.keys())}") 