# Development Guide

This document contains information for developers that need further in-depth information on how to setup and use tools
and learn about programing methods used in development of this project.

If you are looking for a higher level overview over the guiding ideas and structure of this project, please visit the
[contribution guide](Contribution-Guide.md).

## Project Setup and Poetry

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
If you do not want to modify your system installation of Python you can use [conda](https://docs.conda.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv).
However, there are some issues with using conda.
Please, check the [trouble shooting guide](#trouble-shooting) below.

*gaitmap* uses [poetry](https://python-poetry.org) to manage its dependencies.
Once you installed poetry, run the following commands to initialize a virtual env and install all development
dependencies:

```bash
poetry install --no-root
```
This will create a new folder called `.venv` inside your project dir.
It contains the python interpreter and all site packages.
You can point your IDE to this folder to use this version of Python.
For PyCharm you can find information about this 
[here](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html).

**In case you encounter any issues (with this command or any command below), please check the section on
 [trouble shooting](#trouble-shooting)**.
 
To add new dependencies:

```bash
poetry add <package name>

# Or in case of a dev dependency
poetry add --dev <package name>
```

For more commands see the [official documentation](https://python-poetry.org/docs/cli/).

To update dependencies after the `pyproject.toml` file was changed (It is a good idea to run this after a `git pull`):
```bash
poetry install --no-root

# or (see differences below)
poetry update
```

Running `poetry install` will only install packages that are not yet installed. `poetry update` will also check, if 
newer versions of already installed packages exist.

### Trouble Shooting

##### `poetry not found` when using `zsh` as shell

If you have trouble installing `poetry` while using `zsh` as your shell, check this [issue](https://github.com/python-poetry/poetry/issues/507)

##### Installation issues while using `conda`

Setting up `poetry` with `conda` as the main Python version can be a little tricky.
First, make sure that you installed poetry in the [recommended way](https://python-poetry.org/docs/#installation) using 
the PowerShell command.

Then you have 2 options to start using poetry for this package:

1. Using a `conda env` instead of `venv`
    - Create a new conda env (using the required Python version for this project).
    - Activate the environment.
    - Run `poetry install --no-root`. Poetry will 
    [detect that you are already using a conda env](https://github.com/python-poetry/poetry/pull/1432) and will use it, 
    instead of creating a new one.
    - After running the poetry install command you should be able to use poetry without activating the conda env again.
    - Setup your IDE to use the conda env you created
2. Using `conda` python and a `venv`
    - This only works, if your conda **base** env has a Python version supported by the project (>= 3.7)
    - Activate the base env
    - Run `poetry install --no-root`. Poetry will create a new venv in the folder `.venv`, because it detects and
        handles the conda base env 
        [different than other envs](https://github.com/maksbotan/poetry/blob/b1058fc2304ea3e2377af357264abd0e1a791a6a/poetry/utils/env.py#L295).
    - Everything else should work like you are not using conda
    
##### Warning/Error about outdated/missing dependencies in the lock file when running `install` or `update`

This happens when the `pyproject.toml` file was changed either by a git update or by manual editing.
To resolve this issue, run the following and then rerun the command you wanted to run:

```bash
poetry update --lock
``` 

This will synchronise the lock file with the packages listed in `pyproject.toml` 

## Configure your IDE

### Jupyter Lab/Notebooks

While we do not (and will not) use Jupyter Notebooks in gaitmap, it might still be helpful to use Jupyter to debug and
prototype your scientific code.
To set up a Jupyter environment that has gaitmap and all dependencies installed, run the following commands:

```
# poetry isntall including root!
poetry install
poetry run doit register_ipykernel
``` 

After this you can start Jupyter as always, but select "gaitmap" as a kernel when you want to run a notebook.

Remember to use the autoreload extension to make sure that Jupyter reloads gaitmap, when ever you change something in 
the library.
Put this in your first cell of every Jupyter Notebook to activate it:

```jupyter
%load_ext autoreload  # Load the extension
%autoreload 2  # Autoreload all modules
```

### Pycharm
You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to settings->Build,Excecution,Deployment->Console->Python Console in the Starting Script:
```jupyter
%load_ext autoreload
%autoreload 2
```