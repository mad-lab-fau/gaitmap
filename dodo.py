from pathlib import Path
import platform

DOIT_CONFIG = {
    "default_tasks": ["format", "pytest", "lint"],
    "backend": "json",
}

HERE = Path(__file__).parent


def task_format():
    """Reformat all files using lint."""
    return {"actions": [["black", HERE]], "verbosity": 1}


def task_test():
    """Run Pytest with coverage."""
    return {"actions": [["pytest", "--cov=gaitmap"]], "verbosity": 2}


def task_lint():
    """Lint all files with Prospector."""
    return {"actions": [["prospector"]], "verbosity": 1}


def task_docs():
    """Build the html docs using Sphinx."""
    if platform.system() == "Windows":
        return {"actions": [[HERE / "docs/make.bat", "html"]], "verbosity": 2}
    else:
        return {"actions": [["make", "-C", HERE / "docs", "html"]], "verbosity": 2}
