# Agentic Django Backend Generator

**Generate complete Django REST API backends from ERD files in minutes, not hours.**

Takes your ERD JSON → AI agents generate → Production-ready Django backend

## What You Get

- **Complete Django Backend**: Models, Views, Serializers, URLs, Settings
- **Production Ready**: Docker, deployment files, requirements.txt
- **3-5x Faster**: Parallel generation with intelligent caching
- **Smart Review**: Only review files that need attention (80% auto-approved)
- **Error Recovery**: Automatic fallback models and retry logic

---

## Quick Start

### **1. Install**
```bash
pip install openai aiofiles
```

### **2. Choose Your Model & Set API Key**
```bash
# Use any model you want!
export OPENROUTER_API_KEY="your_key_here"  # For 300+ models
export OPENAI_API_KEY="your_key_here"      # For GPT models
export ANTHROPIC_API_KEY="your_key_here"   # For Claude models
export GOOGLE_API_KEY="your_key_here"      # For Gemini models
```

### **3. Run with ANY Model**
```bash
# Free models (no cost!)
python universal_backend_builder.py sample_erd.json --model qwen/qwen3-coder:free
python universal_backend_builder.py sample_erd.json --model deepseek/deepseek-r1-0528:free

# Premium models (high quality)
python universal_backend_builder.py sample_erd.json --model gpt-4
python universal_backend_builder.py sample_erd.json --model claude-3-sonnet
python universal_backend_builder.py sample_erd.json --model gemini-pro

# Or use presets
python universal_backend_builder.py sample_erd.json --preset free      # Free models
python universal_backend_builder.py sample_erd.json --preset premium   # Best quality
python universal_backend_builder.py sample_erd.json --preset balanced  # Good + cheap
```

**That's it!** You'll get a complete Django backend in the `backend/` folder.

### **🤖 See All Supported Models**
```bash
python universal_backend_builder.py --list-models
```
Shows 15+ models including FREE ones and premium options from OpenAI, Anthropic, Google, etc.

## 🏢 **For Very Complex ERDs (Enterprise)**

Got a massive ERD with 50+ entities? Use our enterprise-grade system:

### **🔍 Analyze Complexity First**
```bash
python enterprise_backend_builder.py your_massive_erd.json --analyze-only
```

### **🚀 Intelligent Processing**
```bash
# For massive ERDs (automatically detects complexity)
python enterprise_backend_builder.py massive_erd.json --preset enterprise

# Example output:
# 📊 ERD Analysis Complete:
#    • Entities: 37
#    • Relationships: 60  
#    • Complexity Score: 0.49
#    • Recommended Chunks: 2
#    • 🟡 MEDIUM COMPLEXITY - Chunked processing recommended
```

### **🧠 What Makes It Efficient:**

**✅ Smart Chunking**: Breaks massive ERDs into dependency-aware chunks
**✅ Token Optimization**: Minimizes API calls and costs for huge schemas  
**✅ Memory Management**: Handles 100+ entity ERDs without memory issues
**✅ Parallel Processing**: Generates multiple chunks simultaneously
**✅ Dependency Resolution**: Handles complex relationships and circular deps
**✅ Progressive Generation**: Builds incrementally for massive systems

## 🔧 **Developer Feedback Integration**

Never lose developer input again! Our system learns from your feedback:

### **📝 Inline Comment Processing**
```python
# Just add comments to generated code:
# FIX: make this async
# TODO: add validation  
# IMPROVE: optimize query
# BUG: email should be unique

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()  # BUG: should be unique
```

### **🔄 Interactive Review Workflow**
```bash
python feedback_demo.py

# Example output:
🔍 REVIEWING: MODELS
📄 Generated Code Preview (15/23 lines):
  1 | from django.db import models
  2 | 
  3 | class User(models.Model):
➤ 4 |     # FIX: make this async
  5 |     name = models.CharField(max_length=100)

💬 Inline Comments Found:
   Line 4: FIX - make this async

🎯 What would you like to do?
   [1] ✅ Approve (code looks good)
   [2] ❌ Reject (regenerate)  
   [3] ✏️  Edit (make changes)
   [4] 💬 Add feedback (comment)
   [5] ⏭️  Skip (no action)
```

### **🧠 Learns & Improves Automatically**
- **Pattern Recognition**: Notices you often request async/await
- **Adaptive Prompts**: Future generations include learned patterns
- **Quality Improvement**: Gets better with each interaction
- **Persistent Memory**: Remembers feedback across sessions

## 📝 ERD Format

```json
{
  "entities": [
    {
      "name": "User",
      "fields": [
        {"name": "id", "type": "AutoField", "primary_key": true},
        {"name": "username", "type": "CharField", "max_length": 150},
        {"name": "email", "type": "EmailField"}
      ]
    }
  ]
}
```  
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
graph TD
    A["🎯 Django Requirements<br/>ERD + Business Rules + Auth"] --> B["🎼 DomainOrchestrator<br/>Central coordinator"]
    
    B --> C["🧠 BusinessLogicAgent<br/>Worker specialist"]
    B --> D["🔐 AuthenticationAgent<br/>Worker specialist"]
    B --> E["🚀 APIAgent<br/>Worker specialist"]
    
    C --> F["🔗 DomainExpertHub<br/>Communication center"]
    D --> F
    E --> F
    
    F --> G["📨 Cross-Domain<br/>Consultation"]
    F --> H["🔍 Cross-Domain<br/>Validation"]
    
    G --> I["🤝 Agent Collaboration<br/>Shared insights"]
    H --> I
    
    I --> J["📋 Knowledge Synthesis<br/>Orchestrator combines results"]
    
    J --> K["🎉 Integrated Django Backend<br/>Business Logic + Auth + API"]
    
    %% Feedback loops
    C -.-> D
    D -.-> C
    C -.-> E
    D -.-> E
    
    %% Styling
    classDef orchestrator fill:#e3f2fd
    classDef workers fill:#e8f5e8
    classDef communication fill:#fff3e0
    classDef output fill:#f3e5f5
    
    class B orchestrator
    class C,D,E workers
    class F,G,H,I communication
    class A,J,K output
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
