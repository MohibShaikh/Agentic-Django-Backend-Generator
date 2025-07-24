# Goal
Build a multi-step OpenAI SDK agent system that takes an ERD or structured table schema as input and generates a fully working Django REST API backend.

# Input
The input is a JSON object representing the ERD. For example:

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

# Output

A Django project scaffolded with:

* `models.py` containing the translated models.
* `serializers.py` for each model.
* `views.py` using DRF ViewSets.
* `urls.py` with routers.
* JWT authentication setup in `settings.py`.
* `requirements.txt`, `Dockerfile`, and `Procfile`.
* Optional: a shell script to run migrations and create superuser.

# Constraints

* Use OpenAI SDK and function calling to handle generation.

* Use separate functions for:

  * `generate_models`
  * `generate_serializers`
  * `generate_views`
  * `generate_urls`
  * `generate_auth_setup`
  * `generate_deployment_files`

* Store generated code in appropriate files in a Django project folder (`backend/`).

* Optionally use tools like `os`, `subprocess`, and `pathlib` to write files and run migrations.

# Task

Start by:

1. Initializing a Python script `agent_backend_builder.py`.
2. Write a function to accept the ERD JSON as input.
3. Call the OpenAI GPT function for each step in the pipeline (model → serializer → view → url → auth → deploy).
4. Write each generated file to disk.
5. Print progress at each step.

# Tools

* OpenAI SDK (`openai==1.11+`)
* Python 3.10+
* Optional: `typer` or `argparse` for CLI

# Deliverable

A working CLI script that generates a Django project from ERD input and optionally runs `makemigrations`, `migrate`, and `runserver`. 