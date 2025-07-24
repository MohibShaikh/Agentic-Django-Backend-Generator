import json
import sys
from pathlib import Path
import os
import re
import ast
import tempfile
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import uuid
import textwrap

# Load environment variables from .env file
load_dotenv()

def strip_markdown_code_fence(text):
    # Remove triple backticks and optional language specifier
    return re.sub(r'^```(?:python)?\s*|```$', '', text.strip(), flags=re.MULTILINE).strip()

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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
    def generate(self, erd):
        print("[ModelAgent] Generating models.py from ERD using OpenAI...")
        system_prompt = (
            "Role: You are an expert Django backend engineer specializing in database modeling.\n"
            "Task: Generate a Django models.py file from a provided ERD (Entity-Relationship Diagram) in JSON format.\n"
            "Input: The ERD JSON describing entities, fields, and relationships.\n"
            "Output: A complete, valid Django models.py file implementing all entities and relationships.\n"
            "Constraints: \n"
            "- Use only standard Django ORM features.\n"
            "- Follow Django and PEP8 best practices.\n"
            "- Do not include extra comments or explanations.\n"
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
    def __init__(self, client, extra_headers, extra_body):
        self.client = client
        self.extra_headers = extra_headers
        self.extra_body = extra_body
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
        self.model_agent = ModelAgent(self.client, self.extra_headers, self.extra_body)
        self.serializer_agent = SerializerAgent(self.client, self.extra_headers, self.extra_body)
        self.view_agent = ViewAgent(self.client, self.extra_headers, self.extra_body)
        self.router_agent = RouterAgent(self.client, self.extra_headers, self.extra_body)
        self.auth_agent = AuthAgent(self.client, self.extra_headers, self.extra_body)
        self.deployment_agent = DeploymentAgent(self.client, self.extra_headers, self.extra_body)
        self.critic_agent = CriticAgent()
        self.reviser_agent = ReviserAgent(self.client, self.extra_headers, self.extra_body)
        self.django_critic = DjangoCheckCritic()
        self.custom_feature_agent = CustomFeatureAgent(self.client, self.extra_headers, self.extra_body)

    def run(self):
        print("[Agent] Starting backend generation pipeline (parallel)...")
        tasks = {
            'models.py': lambda: self.safe_generate(self.generate_models, "models.py", filetype="python"),
            'serializers.py': lambda: self.safe_generate(self.generate_serializers, "serializers.py", filetype="python"),
            'views.py': lambda: self.safe_generate(self.generate_views, "views.py", filetype="python"),
            'urls.py': lambda: self.safe_generate(self.generate_urls, "urls.py", filetype="python"),
            'settings.py': lambda: self.safe_generate(self.generate_auth_setup, "settings.py", filetype="python"),
        }
        outputs = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {executor.submit(func): name for name, func in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    print(f"[Agent] {name} generation complete.")
                    outputs[name] = result
                except Exception as exc:
                    print(f"[Agent] {name} generated an exception: {exc}")
                    outputs[name] = f"# ERROR: Failed to generate {name}"
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