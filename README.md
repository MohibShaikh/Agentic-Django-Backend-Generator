# Agentic Django Backend Generator

This project is an **agentic AI system** that automatically generates a production-ready Django REST API backend from an Entity-Relationship Diagram (ERD) or structured table schema (in JSON format). It leverages OpenRouter (and optionally OpenAI) LLMs to generate, review, and revise code in a modular, extensible, and human-in-the-loop (HITL) workflow.

# 🧠 Agentic Django Backend Generator

An **agentic AI system** that builds a full production-ready **Django REST API backend** from a given Entity-Relationship Diagram (ERD) or JSON-based schema.

Powered by **OpenRouter-compatible LLMs** (e.g., Qwen3 Coder, DeepSeek), this system uses multi-agent collaboration and optional human-in-the-loop (HITL) feedback to generate scalable, deployable code — from models to authentication and deployment scripts.

---

## Features

- Multi-Agent Architecture (models, serializers, views, routers, auth, etc.)  
- OpenRouter LLMs like `Qwen3-Coder`, `DeepSeek`, etc.  
- Automated Code Review and Self-Revision Agent  
- Human-in-the-Loop (HITL) Approval on Each File  
- Natural Language-Based Custom Feature Generation  
- Docker-ready: `Dockerfile`, `requirements.txt`, `Procfile`  
- Sequential or Parallel Agent Execution  
- Rate Limit Handling for OpenRouter APIs  

---

## Output Structure

```bash
backend/
├── models.py
├── serializers.py
├── views.py
├── urls.py
├── settings.py
├── auth/
│   └── custom_auth.py
├── requirements.txt
├── Dockerfile
└── Procfile
```

## Agentic Workflow Diagram

Below is a high-level overview of the agentic workflow used to generate your Django backend:

```mermaid
flowchart TD
    subgraph Input
        A["User provides ERD (JSON)"]
    end

    subgraph Orchestration
        B["PlannerAgent<br/>(Orchestrates workflow)"]
    end

    subgraph Generation
        C1["ModelAgent<br/>models.py"]
        C2["SerializerAgent<br/>serializers.py"]
        C3["ViewAgent<br/>views.py"]
        C4["RouterAgent<br/>urls.py"]
        C5["AuthAgent<br/>settings.py (auth)"]
        C6["CustomFeatureAgent<br/>(optional)"]
    end

    subgraph HITL_Review["Human-in-the-Loop Review"]
        D1["Approve/Edit/Skip models.py"]
        D2["Approve/Edit/Skip serializers.py"]
        D3["Approve/Edit/Skip views.py"]
        D4["Approve/Edit/Skip urls.py"]
        D5["Approve/Edit/Skip settings.py"]
        D6["Approve/Edit/Skip custom feature"]
    end

    subgraph Deployment
        E["DeploymentAgent<br/>requirements.txt, Dockerfile, Procfile"]
        F["All code written to backend/ directory"]
    end

    A --> B
    B --> C1 --> D1
    D1 --> C2 --> D2
    D2 --> C3 --> D3
    D3 --> C4 --> D4
    D4 --> C5 --> D5
    D5 --> C6 --> D6
    D6 --> E --> F

    %% Styling
    classDef agent fill:#f9f,stroke:#333,stroke-width:1px;
    classDef hitl fill:#bbf,stroke:#333,stroke-width:1px;
    class B,C1,C2,C3,C4,C5,C6 agent;
    class D1,D2,D3,D4,D5,D6 hitl;
```

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your OpenRouter API key:**
   ```bash
   export OPENROUTER_API_KEY=your-openrouter-key
   # Optionally set HTTP-Referer and X-Title for OpenRouter rankings
   export OPENROUTER_REFERER=https://your-site-url.example.com
   export OPENROUTER_TITLE=YourSiteName
   ```
3. **Run the agentic backend builder:**
   ```bash
   python agent_backend_builder.py sample_erd.json
   ```
4. **Follow the prompts:**
   - Review, approve, edit, or skip each generated file.
   - Optionally add custom features when prompted.

## Example ERD Input
```json
{
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
```

## Models Supported
- [x] Qwen3-Coder (qwen/qwen3-coder:free)
- [x] DeepSeek (deepseek/deepseek-r1-0528:free)
- [x] Any OpenRouter-compatible LLM

## Advanced Features
- Modular agent design for easy extension (add new agents for new features)
- Custom Feature Agent for business-specific needs
- Human-in-the-Loop review for enterprise compliance
- Rate limit handling for OpenRouter models

## License
MIT

---

**Build your Django backend, faster, safer, and with full AI + human control!** 