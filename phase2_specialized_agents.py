"""
Phase 2: Specialized Domain Agents Architecture
===============================================

Building on Phase 1 MVP success, this implements domain-expert agents
that handle specific aspects of Django backend generation.

Key Principles:
- One domain at a time
- Each agent is a true expert in their domain  
- Agents collaborate through structured protocols
- Manageable complexity growth
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid

# ============================================================================
# Phase 2 Domain Types
# ============================================================================

class DomainType(Enum):
    """Specialized domains for expert agents"""
    BUSINESS_LOGIC = "business_logic"
    AUTHENTICATION = "authentication" 
    API_DESIGN = "api_design"
    TESTING = "testing"
    DATABASE = "database"
    DEPLOYMENT = "deployment"

class ExpertiseLevel(Enum):
    """Agent expertise levels"""
    JUNIOR = "junior"
    SENIOR = "senior"
    ARCHITECT = "architect"
    SPECIALIST = "specialist"

# ============================================================================
# Specialized Communication Protocols
# ============================================================================

@dataclass
class DomainRequest:
    """Request from one domain agent to another"""
    from_domain: DomainType
    to_domain: DomainType
    request_type: str  # "consultation", "validation", "generation", "review"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    context: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # "low", "normal", "high", "critical"
    requires_response: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DomainResponse:
    """Response from domain expert"""
    request_id: str
    from_domain: DomainType
    response_type: str  # "approved", "rejected", "modified", "suggestion"
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8  # 0.0 to 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DomainKnowledge:
    """Shared knowledge specific to domains"""
    erd: Dict[str, Any] = field(default_factory=dict)
    business_rules: List[Dict] = field(default_factory=list)
    auth_requirements: Dict[str, Any] = field(default_factory=dict)
    api_specifications: Dict[str, Any] = field(default_factory=dict)
    test_scenarios: List[Dict] = field(default_factory=list)
    generated_artifacts: Dict[str, str] = field(default_factory=dict)
    domain_constraints: Dict[DomainType, List[str]] = field(default_factory=dict)

# ============================================================================
# Domain Expert Hub
# ============================================================================

class DomainExpertHub:
    """Central coordination for domain expert agents"""
    
    def __init__(self):
        self.agents: Dict[DomainType, 'BaseDomainAgent'] = {}
        self.knowledge = DomainKnowledge()
        self.active_requests: Dict[str, DomainRequest] = {}
        self.collaboration_history: List[Dict] = []
        
    def register_agent(self, agent: 'BaseDomainAgent'):
        """Register a domain expert agent"""
        self.agents[agent.domain] = agent
        agent.set_hub(self)
        print(f"🎯 Registered {agent.domain.value} expert: {agent.expertise_level.value}")
        
    async def send_domain_request(self, request: DomainRequest) -> Optional[DomainResponse]:
        """Send request between domain agents"""
        if request.to_domain not in self.agents:
            print(f"❌ No expert available for {request.to_domain.value}")
            return None
            
        self.active_requests[request.id] = request
        target_agent = self.agents[request.to_domain]
        
        print(f"📨 {request.from_domain.value} → {request.to_domain.value}: {request.request_type}")
        response = await target_agent.handle_domain_request(request)
        
        if response:
            self.collaboration_history.append({
                'request': request,
                'response': response,
                'timestamp': datetime.now()
            })
            
        return response
        
    def broadcast_knowledge_update(self, domain: DomainType, update: Dict[str, Any]):
        """Broadcast knowledge updates to relevant agents"""
        print(f"📢 Knowledge update from {domain.value}: {list(update.keys())}")
        for agent in self.agents.values():
            if agent.domain != domain:
                agent.on_knowledge_update(domain, update)
                
    def get_domain_insights(self, requesting_domain: DomainType) -> Dict[str, Any]:
        """Get insights from all domains for decision making"""
        insights = {}
        for domain, agent in self.agents.items():
            if domain != requesting_domain:
                insights[domain.value] = agent.provide_domain_insights()
        return insights

# ============================================================================
# Base Domain Agent
# ============================================================================

class BaseDomainAgent(ABC):
    """Base class for all domain expert agents"""
    
    def __init__(self, domain: DomainType, expertise_level: ExpertiseLevel):
        self.domain = domain
        self.expertise_level = expertise_level
        self.hub: Optional[DomainExpertHub] = None
        self.domain_rules: List[str] = []
        self.capabilities: Set[str] = set()
        self.current_tasks: List[str] = []
        
    def set_hub(self, hub: DomainExpertHub):
        """Connect to the domain expert hub"""
        self.hub = hub
        
    @abstractmethod
    async def generate_domain_code(self, requirements: Dict[str, Any]) -> str:
        """Generate code specific to this domain"""
        pass
        
    @abstractmethod
    async def validate_cross_domain(self, other_domain: DomainType, 
                                  code: str) -> DomainResponse:
        """Validate code from another domain affecting this domain"""
        pass
        
    @abstractmethod
    def provide_domain_insights(self) -> Dict[str, Any]:
        """Provide insights about this domain for other agents"""
        pass
        
    async def handle_domain_request(self, request: DomainRequest) -> DomainResponse:
        """Handle requests from other domain agents"""
        print(f"🤔 {self.domain.value} expert processing {request.request_type}")
        
        if request.request_type == "consultation":
            return await self._handle_consultation(request)
        elif request.request_type == "validation":
            return await self._handle_validation(request)
        elif request.request_type == "generation":
            return await self._handle_generation(request)
        elif request.request_type == "review":
            return await self._handle_review(request)
        else:
            return DomainResponse(
                request_id=request.id,
                from_domain=self.domain,
                response_type="rejected",
                content={"error": f"Unknown request type: {request.request_type}"},
                confidence=0.0
            )
            
    async def _handle_consultation(self, request: DomainRequest) -> DomainResponse:
        """Handle consultation requests"""
        insights = self.provide_domain_insights()
        return DomainResponse(
            request_id=request.id,
            from_domain=self.domain,
            response_type="approved",
            content={"insights": insights, "recommendations": []},
            confidence=0.9,
            reasoning=f"{self.domain.value} expert consultation provided"
        )
        
    async def _handle_validation(self, request: DomainRequest) -> DomainResponse:
        """Handle validation requests"""
        code = request.context.get("code", "")
        is_valid = await self._validate_domain_code(code)
        
        return DomainResponse(
            request_id=request.id,
            from_domain=self.domain,
            response_type="approved" if is_valid else "rejected",
            content={"valid": is_valid, "suggestions": []},
            confidence=0.8,
            reasoning=f"Domain validation {'passed' if is_valid else 'failed'}"
        )
        
    async def _handle_generation(self, request: DomainRequest) -> DomainResponse:
        """Handle code generation requests"""
        try:
            code = await self.generate_domain_code(request.context)
            return DomainResponse(
                request_id=request.id,
                from_domain=self.domain,
                response_type="approved",
                content={"generated_code": code},
                confidence=0.9,
                reasoning=f"Successfully generated {self.domain.value} code"
            )
        except Exception as e:
            return DomainResponse(
                request_id=request.id,
                from_domain=self.domain,
                response_type="rejected",
                content={"error": str(e)},
                confidence=0.0,
                reasoning=f"Generation failed: {str(e)}"
            )
            
    async def _handle_review(self, request: DomainRequest) -> DomainResponse:
        """Handle code review requests"""
        code = request.context.get("code", "")
        review_result = await self._review_domain_code(code)
        
        return DomainResponse(
            request_id=request.id,
            from_domain=self.domain,
            response_type="approved",
            content=review_result,
            confidence=0.8,
            reasoning=f"{self.domain.value} expert review completed"
        )
        
    async def _validate_domain_code(self, code: str) -> bool:
        """Validate code against domain rules"""
        # Basic validation - to be overridden by specific agents
        return len(code.strip()) > 0
        
    async def _review_domain_code(self, code: str) -> Dict[str, Any]:
        """Review code from domain perspective"""
        # Basic review - to be overridden by specific agents
        return {
            "score": 8.0,
            "suggestions": [],
            "warnings": [],
            "approved": True
        }
        
    def on_knowledge_update(self, source_domain: DomainType, update: Dict[str, Any]):
        """React to knowledge updates from other domains"""
        print(f"📝 {self.domain.value} received update from {source_domain.value}")
        
    async def consult_domain(self, target_domain: DomainType, 
                           context: Dict[str, Any]) -> Optional[DomainResponse]:
        """Consult another domain expert"""
        if not self.hub:
            return None
            
        request = DomainRequest(
            from_domain=self.domain,
            to_domain=target_domain,
            request_type="consultation",
            context=context
        )
        
        return await self.hub.send_domain_request(request)

# ============================================================================
# Domain Orchestrator
# ============================================================================

class DomainOrchestrator:
    """Orchestrates collaboration between domain experts"""
    
    def __init__(self):
        self.hub = DomainExpertHub()
        self.generation_pipeline: List[DomainType] = []
        
    def register_agent(self, agent: BaseDomainAgent):
        """Register a domain expert"""
        self.hub.register_agent(agent)
        
    def set_generation_pipeline(self, domains: List[DomainType]):
        """Define the order of domain generation"""
        self.generation_pipeline = domains
        print(f"🔄 Pipeline: {' → '.join(d.value for d in domains)}")
        
    async def orchestrate_generation(self, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Orchestrate code generation across all domains"""
        print(f"\n🚀 Starting domain-expert generation pipeline...")
        results = {}
        
        # Update shared knowledge
        self.hub.knowledge.erd = requirements.get("erd", {})
        self.hub.knowledge.business_rules = requirements.get("business_rules", [])
        self.hub.knowledge.auth_requirements = requirements.get("auth_requirements", {})
        
        # Execute pipeline
        for domain in self.generation_pipeline:
            if domain in self.hub.agents:
                print(f"\n🎯 Executing {domain.value} expert...")
                agent = self.hub.agents[domain]
                
                # Get insights from other domains
                insights = self.hub.get_domain_insights(domain)
                
                # Generate domain-specific code
                context = {
                    "requirements": requirements,
                    "domain_insights": insights,
                    "shared_knowledge": self.hub.knowledge
                }
                
                try:
                    code = await agent.generate_domain_code(context)
                    results[domain.value] = code
                    
                    # Update shared knowledge
                    self.hub.knowledge.generated_artifacts[domain.value] = code
                    self.hub.broadcast_knowledge_update(domain, {"generated": True})
                    
                    print(f"✅ {domain.value} generation complete")
                    
                except Exception as e:
                    print(f"❌ {domain.value} generation failed: {e}")
                    results[domain.value] = f"# Generation failed: {e}"
                    
        return results
        
    async def validate_cross_domain_compatibility(self) -> Dict[str, Any]:
        """Validate that all generated code works together"""
        print(f"\n🔍 Cross-domain validation...")
        validation_results = {}
        
        for domain_a in self.hub.agents:
            for domain_b in self.hub.agents:
                if domain_a != domain_b:
                    agent_a = self.hub.agents[domain_a]
                    code_b = self.hub.knowledge.generated_artifacts.get(domain_b.value, "")
                    
                    if code_b:
                        validation = await agent_a.validate_cross_domain(domain_b, code_b)
                        key = f"{domain_a.value}_validates_{domain_b.value}"
                        validation_results[key] = validation
                        
        return validation_results

# ============================================================================
# Example Usage Setup
# ============================================================================

def create_phase2_demo_requirements():
    """Create demo requirements for Phase 2 testing"""
    return {
        "erd": {
            "entities": {
                "User": {
                    "fields": {"username": "str", "email": "str", "is_active": "bool"},
                    "relationships": {}
                },
                "Product": {
                    "fields": {"name": "str", "price": "decimal", "description": "text"},
                    "relationships": {"user": "ForeignKey(User)"}
                }
            }
        },
        "business_rules": [
            {"rule": "Users can only edit their own products"},
            {"rule": "Price must be greater than 0"},
            {"rule": "Username must be unique"}
        ],
        "auth_requirements": {
            "authentication": "JWT",
            "permissions": ["IsAuthenticated", "IsOwnerOrReadOnly"],
            "roles": ["user", "admin"]
        },
        "api_requirements": {
            "format": "REST",
            "versioning": "v1",
            "pagination": True,
            "filtering": True
        }
    }

if __name__ == "__main__":
    print("🏗️ Phase 2: Specialized Domain Agents Architecture")
    print("Ready for domain expert implementation!") 