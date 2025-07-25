"""
BusinessLogicAgent - Domain Expert for Business Rules & Validation
================================================================

First specialized agent in Phase 2: Handles complex business logic,
validation rules, custom model methods, and business workflows.

Domain Expertise:
- Business rule implementation
- Custom model methods and properties  
- Data validation and constraints
- Business workflow logic
- Cross-model business relationships
- Custom managers and querysets
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from phase2_specialized_agents import (
    BaseDomainAgent, DomainType, ExpertiseLevel, DomainResponse, DomainRequest
)

class BusinessLogicAgent(BaseDomainAgent):
    """Expert agent for Django business logic implementation"""
    
    def __init__(self, expertise_level: ExpertiseLevel = ExpertiseLevel.SENIOR):
        super().__init__(DomainType.BUSINESS_LOGIC, expertise_level)
        
        # Domain-specific capabilities
        self.capabilities = {
            "custom_model_methods",
            "business_rule_validation", 
            "custom_managers",
            "computed_properties",
            "business_workflow_logic",
            "cross_model_constraints",
            "data_integrity_rules",
            "custom_querysets"
        }
        
        # Business logic patterns we can implement
        self.business_patterns = {
            "ownership_validation": "Validate user owns resource",
            "state_transitions": "Manage object state changes", 
            "calculated_fields": "Compute derived values",
            "business_constraints": "Enforce business rules",
            "audit_logging": "Track business operations",
            "soft_delete": "Logical deletion patterns",
            "hierarchical_data": "Tree/nested structures",
            "workflow_states": "Multi-step processes"
        }
        
        self.client = None
        
    def set_openai_client(self, client: Any):
        """Set OpenAI client for AI-powered generation"""
        self.client = client
        
    async def generate_domain_code(self, requirements: Dict[str, Any]) -> str:
        """Generate business logic code for Django models"""
        print(f"🧠 BusinessLogicAgent generating code...")
        
        shared_knowledge = requirements.get("shared_knowledge", {})
        if hasattr(shared_knowledge, 'erd'):
            erd = shared_knowledge.erd
        else:
            erd = shared_knowledge.get("erd", {}) if isinstance(shared_knowledge, dict) else requirements.get("erd", {})
        business_rules = requirements.get("business_rules", [])
        domain_insights = requirements.get("domain_insights", {})
        
        # Analyze business requirements
        analysis = await self._analyze_business_requirements(erd, business_rules)
        
        # Generate business logic code
        if self.client:
            code = await self._generate_with_ai(erd, business_rules, analysis, domain_insights)
        else:
            code = await self._generate_without_ai(erd, business_rules, analysis)
            
        return code
        
    async def _analyze_business_requirements(self, erd: Dict, rules: List[Dict]) -> Dict[str, Any]:
        """Analyze ERD and business rules to determine required logic"""
        analysis = {
            "custom_methods_needed": [],
            "validation_rules": [],
            "computed_properties": [],
            "custom_managers": [],
            "business_constraints": [],
            "workflow_states": []
        }
        
        # Analyze entities for business logic needs
        entities = erd.get("entities", {})
        for entity_name, entity_data in entities.items():
            fields = entity_data.get("fields", {})
            
            # Check for price fields -> need validation
            if any("price" in field.lower() for field in fields):
                analysis["validation_rules"].append({
                    "entity": entity_name,
                    "rule": "price_validation",
                    "description": "Ensure price is positive"
                })
                
            # Check for status/state fields -> need state management
            if any(field.lower() in ["status", "state"] for field in fields):
                analysis["workflow_states"].append({
                    "entity": entity_name,
                    "field": next(f for f in fields if f.lower() in ["status", "state"]),
                    "description": "State transition logic needed"
                })
                
            # Check for user relationships -> need ownership validation
            relationships = entity_data.get("relationships", {})
            if any("user" in rel.lower() for rel in relationships.values()):
                analysis["custom_methods_needed"].append({
                    "entity": entity_name,
                    "method": "is_owned_by",
                    "description": "Check if user owns this resource"
                })
                
        # Analyze explicit business rules
        for rule in rules:
            rule_text = rule.get("rule", "").lower()
            
            if "own" in rule_text or "owner" in rule_text:
                analysis["business_constraints"].append({
                    "type": "ownership",
                    "description": rule.get("rule", ""),
                    "implementation": "Add ownership validation methods"
                })
                
            if "greater than" in rule_text or "positive" in rule_text:
                analysis["validation_rules"].append({
                    "type": "numeric_validation", 
                    "description": rule.get("rule", ""),
                    "implementation": "Add custom validation in clean method"
                })
                
            if "unique" in rule_text:
                analysis["business_constraints"].append({
                    "type": "uniqueness",
                    "description": rule.get("rule", ""),
                    "implementation": "Add unique constraint validation"
                })
                
        return analysis
        
    async def _generate_with_ai(self, erd: Dict, rules: List[Dict], 
                               analysis: Dict, insights: Dict) -> str:
        """Generate business logic using AI"""
        
        prompt = self._create_business_logic_prompt(erd, rules, analysis, insights)
        
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
            return await self._generate_without_ai(erd, rules, analysis)
            
    def _create_business_logic_prompt(self, erd: Dict, rules: List[Dict], 
                                    analysis: Dict, insights: Dict) -> str:
        """Create structured prompt for business logic generation"""
        
        entities = list(erd.get("entities", {}).keys())[:2]
        rules_text = [r.get("rule", "")[:30] for r in rules[:2]]
        
        return f"""Generate Django business logic for {entities}. Rules: {rules_text}. Return Python code with custom methods and validation."""
        
    async def _generate_without_ai(self, erd: Dict, rules: List[Dict], 
                                 analysis: Dict) -> str:
        """Generate basic business logic without AI"""
        
        code_parts = []
        code_parts.append("# Business Logic Extensions")
        code_parts.append("# Generated by BusinessLogicAgent")
        code_parts.append("")
        code_parts.append("from django.core.exceptions import ValidationError")
        code_parts.append("from django.db import models")
        code_parts.append("from decimal import Decimal")
        code_parts.append("")
        
        # Generate custom managers
        if analysis.get("custom_managers"):
            code_parts.append("# Custom Managers")
            for manager in analysis["custom_managers"]:
                code_parts.append(f"class {manager['entity']}Manager(models.Manager):")
                code_parts.append('    """Custom manager for business queries"""')
                code_parts.append("    def active(self):")
                code_parts.append("        return self.filter(is_active=True)")
                code_parts.append("")
                
        # Generate business logic mixins
        entities = erd.get("entities", {})
        for entity_name, entity_data in entities.items():
            
            # Check if this entity needs business logic
            needs_logic = any(
                entity_name in str(item) for analysis_list in analysis.values() 
                for item in (analysis_list if isinstance(analysis_list, list) else [])
            )
            
            if needs_logic:
                code_parts.append(f"# Business Logic for {entity_name}")
                code_parts.append(f"class {entity_name}BusinessLogic:")
                code_parts.append(f'    """Business logic mixin for {entity_name} model"""')
                code_parts.append("")
                
                # Add ownership validation if needed
                ownership_methods = [m for m in analysis.get("custom_methods_needed", []) 
                                   if m.get("entity") == entity_name and "owned" in m.get("method", "")]
                if ownership_methods:
                    code_parts.append("    def is_owned_by(self, user):")
                    code_parts.append('        """Check if user owns this resource"""')
                    code_parts.append("        return hasattr(self, 'user') and self.user == user")
                    code_parts.append("")
                    
                # Add validation methods
                validation_rules = [v for v in analysis.get("validation_rules", [])
                                  if v.get("entity") == entity_name]
                if validation_rules:
                    code_parts.append("    def clean(self):")
                    code_parts.append('        """Custom validation for business rules"""')
                    code_parts.append("        super().clean()")
                    
                    for rule in validation_rules:
                        if rule.get("rule") == "price_validation":
                            code_parts.append("        if hasattr(self, 'price') and self.price <= 0:")
                            code_parts.append("            raise ValidationError('Price must be greater than 0')")
                            
                    code_parts.append("")
                    
                # Add computed properties
                fields = entity_data.get("fields", {})
                if "price" in fields:
                    code_parts.append("    @property")
                    code_parts.append("    def price_with_tax(self):")
                    code_parts.append('        """Calculate price including tax"""')
                    code_parts.append("        return self.price * Decimal('1.1')  # 10% tax")
                    code_parts.append("")
                    
                code_parts.append("")
                
        # Add usage instructions
        code_parts.append("# Usage Instructions:")
        code_parts.append("# 1. Add these mixins to your Django models")
        code_parts.append("# 2. Inherit from BusinessLogic classes in your models")
        code_parts.append("# 3. Call clean() in model save methods")
        code_parts.append("# 4. Use custom managers for business queries")
        
        return "\n".join(code_parts)
        
    async def validate_cross_domain(self, other_domain: DomainType, code: str) -> DomainResponse:
        """Validate how other domain code affects business logic"""
        
        concerns = []
        suggestions = []
        
        if other_domain == DomainType.AUTHENTICATION:
            # Check if auth code considers business ownership rules
            if "user" in code.lower() and "owner" not in code.lower():
                concerns.append("Authentication should consider ownership validation")
                suggestions.append("Add ownership checks in permission classes")
                
        elif other_domain == DomainType.API_DESIGN:
            # Check if API respects business validation
            if "serializer" in code.lower() and "clean" not in code.lower():
                concerns.append("API should trigger model validation")
                suggestions.append("Call model.clean() in serializer validation")
                
        return DomainResponse(
            request_id="validation",
            from_domain=self.domain,
            response_type="approved" if not concerns else "suggestion",
            content={
                "concerns": concerns,
                "suggestions": suggestions,
                "compatibility_score": 0.9 if not concerns else 0.7
            },
            confidence=0.8,
            reasoning=f"Business logic validation for {other_domain.value}"
        )
        
    def provide_domain_insights(self) -> Dict[str, Any]:
        """Provide business logic insights for other agents"""
        return {
            "business_patterns": list(self.business_patterns.keys()),
            "validation_points": [
                "Model clean methods",
                "Custom validation rules", 
                "Business constraints",
                "Ownership validation"
            ],
            "integration_needs": {
                "authentication": "Needs ownership validation methods",
                "api_design": "Needs business rule enforcement", 
                "testing": "Needs business logic test scenarios"
            },
            "business_rules_implemented": True,
            "custom_methods_available": True
        }
        
    async def consult_for_complex_rules(self, rules: List[str]) -> Dict[str, Any]:
        """Specialized consultation for complex business rules"""
        print(f"🧠 BusinessLogicAgent analyzing complex rules...")
        
        complexity_analysis = {}
        
        for rule in rules:
            if any(keyword in rule.lower() for keyword in ["workflow", "state", "transition"]):
                complexity_analysis[rule] = {
                    "type": "state_machine",
                    "complexity": "high",
                    "recommendation": "Implement with django-fsm or custom state management"
                }
            elif any(keyword in rule.lower() for keyword in ["calculate", "compute", "derive"]):
                complexity_analysis[rule] = {
                    "type": "computed_field",
                    "complexity": "medium", 
                    "recommendation": "Use model properties or custom methods"
                }
            else:
                complexity_analysis[rule] = {
                    "type": "validation",
                    "complexity": "low",
                    "recommendation": "Implement in model clean() method"
                }
                
        return {
            "total_rules": len(rules),
            "complexity_breakdown": complexity_analysis,
            "implementation_strategy": "Implement in order: validation -> computed -> state_machine",
            "estimated_effort": "Medium"
        }

# ============================================================================
# Integration Helper
# ============================================================================

def create_business_logic_agent(openai_client: Optional[Any] = None) -> BusinessLogicAgent:
    """Factory function to create configured BusinessLogicAgent"""
    agent = BusinessLogicAgent(ExpertiseLevel.SENIOR)
    
    if openai_client:
        agent.set_openai_client(openai_client)
        
    print(f"✅ BusinessLogicAgent created with {len(agent.capabilities)} capabilities")
    return agent

if __name__ == "__main__":
    # Demo usage
    agent = create_business_logic_agent()
    print(f"🧠 BusinessLogicAgent ready")
    print(f"📋 Capabilities: {', '.join(agent.capabilities)}")
    print(f"🎯 Patterns: {', '.join(agent.business_patterns.keys())}") 