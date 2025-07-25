#!/usr/bin/env python3
"""
Django Backend Generator - Production CLI
==========================================

Multi-Agent Django Backend Generator using Orchestrator-workers pattern.
Users run this script to generate complete Django backends from ERD files.

Usage:
    python generate_django_backend.py --erd my_project.json
    python generate_django_backend.py --erd ecommerce.json --output ./my_project/
    python generate_django_backend.py --help
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env")
except ImportError:
    print("⚠️  python-dotenv not found. Install with: pip install python-dotenv")

# Import our multi-agent system
from phase2_specialized_agents import DomainOrchestrator, DomainType
from business_logic_agent import create_business_logic_agent
from authentication_agent import create_authentication_agent
from api_agent import create_api_agent
from testing_agent import create_testing_agent

# OpenRouter LLM integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI package not found. Install with: pip install openai")

def load_erd_file(erd_path: str) -> dict:
    """Load ERD from JSON file"""
    try:
        with open(erd_path, 'r') as f:
            erd_data = json.load(f)
        print(f"Loaded ERD: {erd_path}")
        return erd_data
    except FileNotFoundError:
        print(f"ERD file not found: {erd_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in ERD file: {e}")
        sys.exit(1)

def validate_erd(erd_data: dict) -> bool:
    """Validate ERD structure"""
    required_keys = ['erd']
    
    for key in required_keys:
        if key not in erd_data:
            print(f"Missing required key in ERD: {key}")
            return False
            
    if 'entities' not in erd_data['erd']:
        print(f"ERD must contain 'entities'")
        return False
        
    print(f"ERD validation passed")
    return True

def setup_output_directory(output_path: str) -> Path:
    """Create output directory for generated files"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    return output_dir

def display_erd_summary(erd_data: dict):
    """Display summary of loaded ERD"""
    entities = erd_data['erd']['entities']
    business_rules = erd_data.get('business_rules', [])
    auth_reqs = erd_data.get('auth_requirements', {})
    
    print(f"\nERD Summary:")
    print(f"   Entities: {list(entities.keys())}")
    print(f"   Business Rules: {len(business_rules)}")
    
    if auth_reqs:
        print(f"   Authentication: {auth_reqs.get('authentication', 'None')}")
        print(f"   Roles: {auth_reqs.get('roles', [])}")
    else:
        print(f"   Authentication: Default Django auth")

def setup_openrouter_client() -> Optional[Any]:
    """Set up OpenRouter client for LLM-powered generation"""
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI package not available - using template generation")
        return None
        
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not found - using template generation")
        return None
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://django-generator.com"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "Django Multi-Agent Generator")
            }
        )
        print("OpenRouter LLM client initialized")
        return client
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenRouter client: {e}")
        return None

async def generate_django_backend(erd_data: dict, output_dir: Path) -> dict:
    """Generate complete Django backend using multi-agent system"""
    print(f"\nStarting Django Backend Generation...")
    print(f"Initializing Orchestrator-workers Pattern...")
    
    # Set up LLM client for agents
    openai_client = setup_openrouter_client()
    
    # Create orchestrator
    orchestrator = DomainOrchestrator()
    
    # Register domain expert agents with LLM capability
    business_agent = create_business_logic_agent(openai_client)
    auth_agent = create_authentication_agent(openai_client)
    api_agent = create_api_agent(openai_client)
    testing_agent = create_testing_agent(openai_client)
    
    orchestrator.register_agent(business_agent)
    orchestrator.register_agent(auth_agent)
    orchestrator.register_agent(api_agent)
    orchestrator.register_agent(testing_agent)
    
    # Set generation pipeline - logical order of dependencies
    orchestrator.set_generation_pipeline([
        DomainType.BUSINESS_LOGIC,
        DomainType.AUTHENTICATION,
        DomainType.API_DESIGN,
        DomainType.TESTING
    ])
    
    print(f"Running multi-agent generation...")
    
    # Generate code
    results = await orchestrator.orchestrate_generation(erd_data)
    
    print(f"Code generation complete!")
    return results

def generate_models_from_erd(erd_data: dict) -> str:
    """Generate Django models.py from ERD"""
    entities = erd_data['erd']['entities']
    
    code_lines = [
        "# Generated Django Models",
        "from django.db import models",
        "from django.contrib.auth.models import AbstractUser",
        "from django.core.exceptions import ValidationError",
        "",
    ]
    
    # Generate User model if not present
    if 'User' not in entities:
        code_lines.extend([
            "class User(AbstractUser):",
            '    """Extended user model"""',
            "    pass",
            "",
        ])
    
    # Generate models for each entity
    for entity_name, entity_data in entities.items():
        if entity_name == 'User':
            code_lines.extend([
                "class User(AbstractUser):",
                '    """Extended user model"""',
            ])
        else:
            code_lines.extend([
                f"class {entity_name}(models.Model):",
                f'    """Generated {entity_name} model"""',
            ])
        
        # Add fields
        fields = entity_data.get('fields', {})
        for field_name, field_type in fields.items():
            if field_name in ['id', 'created_at', 'updated_at'] and entity_name != 'User':
                continue  # Skip auto fields
                
            if field_type == 'str':
                code_lines.append(f"    {field_name} = models.CharField(max_length=200)")
            elif field_type == 'text':
                code_lines.append(f"    {field_name} = models.TextField()")
            elif field_type == 'int':
                code_lines.append(f"    {field_name} = models.IntegerField()")
            elif field_type == 'bool':
                code_lines.append(f"    {field_name} = models.BooleanField(default=False)")
            elif field_type == 'datetime':
                if field_name == 'created_at':
                    code_lines.append(f"    {field_name} = models.DateTimeField(auto_now_add=True)")
                elif field_name == 'updated_at':
                    code_lines.append(f"    {field_name} = models.DateTimeField(auto_now=True)")
                else:
                    code_lines.append(f"    {field_name} = models.DateTimeField()")
            elif field_type == 'decimal':
                code_lines.append(f"    {field_name} = models.DecimalField(max_digits=10, decimal_places=2)")
        
        # Add relationships
        relationships = entity_data.get('relationships', {})
        for rel_name, rel_def in relationships.items():
            if 'ForeignKey' in rel_def:
                target = rel_def.replace('ForeignKey(', '').replace(')', '')
                code_lines.append(f"    {rel_name} = models.ForeignKey({target}, on_delete=models.CASCADE, related_name='{entity_name.lower()}s')")
            elif 'ManyToMany' in rel_def:
                target = rel_def.replace('ManyToMany(', '').replace(')', '')
                code_lines.append(f"    {rel_name} = models.ManyToManyField({target}, related_name='{entity_name.lower()}s')")
        
        code_lines.extend([
            "",
            "    def __str__(self):",
            f"        return f'{{{entity_name} {{self.pk}}'",
            "",
        ])
    
    return "\n".join(code_lines)

def create_django_project_structure(output_dir: Path, project_name: str = "core", app_name: str = "backend"):
    """Create proper Django project structure with separate core and app directories"""
    # Create core project directory (contains settings, urls, wsgi, etc.)
    core_dir = output_dir / project_name
    core_dir.mkdir(exist_ok=True)
    
    # Create app directory 
    app_dir = output_dir / app_name
    app_dir.mkdir(exist_ok=True)
    
    # Create migrations directory for the app
    migrations_dir = app_dir / "migrations"
    migrations_dir.mkdir(exist_ok=True)
    
    return core_dir, app_dir, migrations_dir

def save_generated_files(results: dict, erd_data: dict, output_dir: Path):
    """Save all generated files in proper Django project structure"""
    print(f"\nCreating Django project structure...")
    
    # Create Django project structure with separate core and app directories
    core_dir, app_dir, migrations_dir = create_django_project_structure(output_dir, "core", "backend")
    
    # Generate app files
    print(f"Creating Django app: backend/")
    
    # 1. __init__.py
    init_path = app_dir / "__init__.py"
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write("")
    print(f"   ✅ backend/__init__.py")
    
    # 2. apps.py
    apps_content = '''from django.apps import AppConfig


class BackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend'
'''
    apps_path = app_dir / "apps.py"
    with open(apps_path, 'w', encoding='utf-8') as f:
        f.write(apps_content)
    print(f"   ✅ backend/apps.py")
    
    # 3. models.py
    models_code = generate_models_from_erd(erd_data)
    models_path = app_dir / "models.py"
    with open(models_path, 'w', encoding='utf-8') as f:
        f.write(models_code)
    print(f"   ✅ backend/models.py ({len(models_code)} characters)")
    
    # 4. admin.py
    admin_content = '''from django.contrib import admin
from .models import *

# Register your models here.
'''
    # Add admin registrations for each model
    entities = erd_data['erd']['entities']
    for entity_name in entities.keys():
        admin_content += f"admin.site.register({entity_name})\n"
    
    admin_path = app_dir / "admin.py"
    with open(admin_path, 'w', encoding='utf-8') as f:
        f.write(admin_content)
    print(f"   ✅ backend/admin.py")
    
    # 5. views.py with DRF ViewSets
    views_content = '''from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import *
from .serializers import *

'''
    
    # Generate ViewSets for each model
    for entity_name in entities.keys():
        views_content += f'''
class {entity_name}ViewSet(viewsets.ModelViewSet):
    """ViewSet for {entity_name} model"""
    queryset = {entity_name}.objects.all()
    serializer_class = {entity_name}Serializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_permissions(self):
        """Set permissions based on action"""
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            self.permission_classes = [permissions.IsAuthenticated]
        return super().get_permissions()
'''
    
    views_path = app_dir / "views.py"
    with open(views_path, 'w') as f:
        f.write(views_content)
    print(f"   ✅ backend/views.py")
    
    # 6. serializers.py
    serializers_content = '''from rest_framework import serializers
from .models import *

'''
    
    # Generate serializers for each model
    for entity_name, entity_data in entities.items():
        fields = list(entity_data.get('fields', {}).keys())
        relationships = list(entity_data.get('relationships', {}).keys())
        all_fields = fields + relationships
        
        serializers_content += f'''
class {entity_name}Serializer(serializers.ModelSerializer):
    """Serializer for {entity_name} model"""
    
    class Meta:
        model = {entity_name}
        fields = {all_fields}
        read_only_fields = ['id', 'created_at', 'updated_at']
'''
    
    serializers_path = app_dir / "serializers.py"
    with open(serializers_path, 'w') as f:
        f.write(serializers_content)
    print(f"   ✅ backend/serializers.py")
    
    # 7. urls.py
    urls_content = '''from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
'''
    
    for entity_name in entities.keys():
        urls_content += f"router.register(r'{entity_name.lower()}s', views.{entity_name}ViewSet)\n"
    
    urls_content += '''
urlpatterns = [
    path('', include(router.urls)),
]
'''
    
    urls_path = app_dir / "urls.py"
    with open(urls_path, 'w') as f:
        f.write(urls_content)
    print(f"   ✅ backend/urls.py")
    
    # 8. tests.py
    tests_content = '''from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase
from rest_framework import status
from .models import *

User = get_user_model()


class ModelTestCase(TestCase):
    """Test models"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_user_creation(self):
        """Test user model creation"""
        self.assertEqual(self.user.username, 'testuser')
        self.assertEqual(self.user.email, 'test@example.com')


class APITestCase(APITestCase):
    """Test API endpoints"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com', 
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_api_endpoints(self):
        """Test API endpoints are accessible"""
        response = self.client.get('/api/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
'''
    
    tests_path = app_dir / "tests.py"
    with open(tests_path, 'w') as f:
        f.write(tests_content)
    print(f"   ✅ backend/tests.py")
    
    # 9. migrations/__init__.py
    migrations_init = migrations_dir / "__init__.py"
    with open(migrations_init, 'w') as f:
        f.write("")
    print(f"   ✅ backend/migrations/__init__.py")
    
    # ========================================================================
    # CREATE CORE PROJECT FILES (Django Configuration)
    # ========================================================================
    print(f"📁 Creating Django core project: core/")
    
    # 10. core/__init__.py
    core_init_path = core_dir / "__init__.py"
    with open(core_init_path, 'w', encoding='utf-8') as f:
        f.write("")
    print(f"   ✅ core/__init__.py")
    
    # 11. core/settings.py
    settings_content = f'''"""
Django Settings for Generated Backend Project
"""

import os
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here-change-in-production'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework_simplejwt',
    'django_filters',
    'backend',  # Your generated app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {{
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        }},
    }},
]

WSGI_APPLICATION = 'core.wsgi.application'

DATABASES = {{
    'default': {{
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }}
}}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {{
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    }},
    {{
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    }},
    {{
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    }},
    {{
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    }},
]

# REST Framework Configuration
REST_FRAMEWORK = {{
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}}

# JWT Configuration
SIMPLE_JWT = {{
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}}

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
'''
    
    settings_path = core_dir / "settings.py"
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write(settings_content)
    print(f"   ✅ core/settings.py")
    
    # 12. core/urls.py
    core_urls_content = '''"""
Main URL Configuration for Django Project
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('backend.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
'''
    
    core_urls_path = core_dir / "urls.py"
    with open(core_urls_path, 'w', encoding='utf-8') as f:
        f.write(core_urls_content)
    print(f"   ✅ core/urls.py")
    
    # 13. core/wsgi.py
    wsgi_content = '''"""
WSGI config for Django project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = get_wsgi_application()
'''
    
    wsgi_path = core_dir / "wsgi.py"
    with open(wsgi_path, 'w', encoding='utf-8') as f:
        f.write(wsgi_content)
    print(f"   ✅ core/wsgi.py")
    
    # 14. core/asgi.py
    asgi_content = '''"""
ASGI config for Django project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = get_asgi_application()
'''
    
    asgi_path = core_dir / "asgi.py"
    with open(asgi_path, 'w', encoding='utf-8') as f:
        f.write(asgi_content)
    print(f"   ✅ core/asgi.py")
    
    # ========================================================================
    # SAVE AGENT-GENERATED FILES
    # ========================================================================
    
    # 15. Save business logic extensions
    if 'business_logic' in results:
        business_path = app_dir / "business_logic.py"
        with open(business_path, 'w') as f:
            f.write(results['business_logic'])
        print(f"   ✅ backend/business_logic.py ({len(results['business_logic'])} characters)")
    
    # 16. Save authentication
    if 'authentication' in results:
        auth_path = app_dir / "authentication.py"
        with open(auth_path, 'w') as f:
            f.write(results['authentication'])
        print(f"   ✅ backend/authentication.py ({len(results['authentication'])} characters)")
    
    # 17. Save API design extensions
    if 'api_design' in results:
        api_path = app_dir / "api_extensions.py"
        with open(api_path, 'w') as f:
            f.write(results['api_design'])
        print(f"   ✅ backend/api_extensions.py ({len(results['api_design'])} characters)")
    
    # 18. Override tests.py with comprehensive test suite
    if 'testing' in results:
        test_path = app_dir / "tests.py"
        with open(test_path, 'w') as f:
            f.write(results['testing'])
        print(f"   ✅ backend/tests.py (comprehensive test suite - {len(results['testing'])} characters)")
    
    # ========================================================================
    # PROJECT ROOT FILES  
    # ========================================================================
    
    # 19. Generate requirements.txt (in root)
    requirements = [
        "Django>=4.2.0",
        "djangorestframework>=3.14.0", 
        "djangorestframework-simplejwt>=5.2.0",
        "django-filter>=23.2",
        "factory-boy>=3.3.0",
        "pytest>=7.4.0",
        "pytest-django>=4.5.2",
        "coverage>=7.2.0",
    ]
    
    req_path = output_dir / "requirements.txt"
    with open(req_path, 'w') as f:
        f.write("\n".join(requirements))
    print(f"   ✅ requirements.txt ({len(requirements)} packages)")
    
    # 20. Generate manage.py
    manage_content = '''#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
'''
    
    manage_path = output_dir / "manage.py"
    with open(manage_path, 'w') as f:
        f.write(manage_content)
    print(f"   ✅ manage.py")
    
    # 21. Generate README.md
    readme_content = f"""# Generated Django Backend

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using Django Multi-Agent Backend Generator

## Django Project Structure (Standard Best Practices):
```
generated_backend/           # Project root
├── manage.py               # Django management script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── core/                   # Django project configuration
│   ├── __init__.py
│   ├── settings.py         # Django settings
│   ├── urls.py             # Main URL configuration  
│   ├── wsgi.py             # WSGI config for deployment
│   └── asgi.py             # ASGI config for async/WebSocket
└── backend/                # Your Django application
    ├── __init__.py
    ├── apps.py             # App configuration
    ├── models.py           # Your ERD models
    ├── views.py            # DRF ViewSets
    ├── serializers.py      # DRF Serializers
    ├── urls.py             # App URL patterns
    ├── admin.py            # Django admin interface
    ├── tests.py            # Comprehensive test suite
    ├── business_logic.py   # Business logic extensions
    ├── authentication.py   # JWT auth & permissions
    ├── api_extensions.py   # Advanced API features
    └── migrations/         # Database migrations
        └── __init__.py
```

## Setup Instructions:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Django Setup
```bash
# Create and run migrations
python manage.py makemigrations backend
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

### 3. API Endpoints
Your API will be available at:
- Admin Interface: http://localhost:8000/admin/
- API Root: http://localhost:8000/api/
"""
    
    # Add endpoint list for each entity
    for entity_name in entities.keys():
        endpoint = entity_name.lower() + 's'
        readme_content += f"- {entity_name}: http://localhost:8000/api/{endpoint}/\n"
    
    readme_content += f"""
## Generated Features:
- ✅ Django models from your ERD
- ✅ REST API with DRF ViewSets
- ✅ JWT Authentication
- ✅ Admin interface
- ✅ Business logic extensions
- ✅ Permission classes
- ✅ Test cases
- ✅ Proper Django app structure

## Usage:
1. Start with `python manage.py runserver`
2. Access admin at `/admin/` 
3. Use API endpoints for your frontend
4. Customize business logic in `backend/business_logic.py`
5. Extend authentication in `backend/authentication.py`
6. Enhance API features in `backend/api_extensions.py`
7. Run comprehensive tests with `python manage.py test`

Generated by Django Multi-Agent Backend Generator
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   ✅ README.md")

def display_success_summary(output_dir: Path, results: dict):
    """Display final success summary"""
    print(f"\n" + "=" * 80)
    print(f"🎉 DJANGO BACKEND GENERATION COMPLETE!")
    print(f"📁 Output Directory: {output_dir.absolute()}")
    print(f"📊 Django App Structure Created:")
    
    # Show the Django app structure
    app_dir = output_dir / "backend"
    print(f"   📁 backend/ (Django app)")
    for file_path in app_dir.glob("*"):
        if file_path.is_file():
            print(f"      📄 {file_path.name}")
        elif file_path.is_dir():
            print(f"      📁 {file_path.name}/")
    
    print(f"   📄 manage.py (Django management)")
    print(f"   📄 settings.py (Django settings)")
    print(f"   📄 urls.py (Main URL config)")
    print(f"   📄 requirements.txt (Dependencies)")
    print(f"   📄 README.md (Setup instructions)")
    
    print(f"\n🚀 Quick Start:")
    print(f"   1. cd {output_dir.name}")
    print(f"   2. pip install -r requirements.txt")
    print(f"   3. python manage.py makemigrations backend")
    print(f"   4. python manage.py migrate")
    print(f"   5. python manage.py runserver")
    
    print(f"\n✨ Your complete Django backend is ready!")
    print(f"🌐 Access your API at: http://localhost:8000/api/")
    print(f"⚙️  Admin interface: http://localhost:8000/admin/")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Generate Django backend from ERD using multi-agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_django_backend.py --erd blog.json
  python generate_django_backend.py --erd ecommerce.json --output ./backend/
  python generate_django_backend.py --erd crm.json --output ../generated_project/
        """
    )
    
    parser.add_argument(
        '--erd', 
        required=True,
        help='Path to ERD JSON file'
    )
    
    parser.add_argument(
        '--output',
        default='./generated_backend/',
        help='Output directory for generated files (default: ./generated_backend/)'
    )
    
    args = parser.parse_args()
    
    print("🏗️ Django Multi-Agent Backend Generator")
    print("=" * 60)
    
    # Load and validate ERD
    erd_data = load_erd_file(args.erd)
    if not validate_erd(erd_data):
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    
    # Display ERD summary
    display_erd_summary(erd_data)
    
    # Generate Django backend
    results = await generate_django_backend(erd_data, output_dir)
    
    # Save generated files
    save_generated_files(results, erd_data, output_dir)
    
    # Display success summary
    display_success_summary(output_dir, results)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n❌ Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 