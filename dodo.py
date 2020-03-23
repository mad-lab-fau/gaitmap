from pathlib import Path
import platform

DOIT_CONFIG = {"default_tasks": ["format", "pytest", "lint"]}

HERE = Path(__file__).parent


def task_format():
    return {"actions": [["black", HERE]], "verbosity": 1}


def task_pytest():
    return {"actions": [["pytest", "--cov=gaitmap"]], "verbosity": 2}


def task_lint():
    return {"actions": [["prospector"]], "verbosity": 1}


def task_docs():
    if platform.system() == "Windows":
        return {"actions": [[HERE / "docs/make.bat", "html"]], "verbosity": 2}
    else:
        return {"actions": [["make", "-C", HERE / "docs", "html"]], "verbosity": 2}
