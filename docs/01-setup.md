# Step 1: Setting up the project with uv

[uv](https://docs.astral.sh/uv/) manages Python projects: it creates an
isolated environment for the project and installs the libraries it needs, so
nothing mixes with the rest of the computer.

Install it once per machine:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the project from inside its folder:

```bash
uv init --bare
```

This creates a single file, `pyproject.toml`, which describes the project:

```toml
[project]
name = "machine_learning_from_scratch"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
```

Commands used throughout the project:

| Command | What it does |
|---|---|
| `uv run <command>` | Runs a command inside the project's environment |
| `uv add <library>` | Adds a library to the project and installs it |

---

[Step 2: Workspace structure →](02-workspace.md)
