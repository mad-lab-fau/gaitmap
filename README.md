# gaitmap - The Gait and Movement Analysis Package

*gaitmap* provides a set of algorithms to analyse your IMU movement data without getting into your way.
It's API is designed to mimic `sklearn` to provide you an familiar and elegant interface


[![pipeline status](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/badges/master/pipeline.svg)](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/commits/master)
[![coverage report](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/badges/master/coverage.svg)](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/commits/master)
[![docs](https://img.shields.io/badge/docs-online-green.svg)](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/README.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Getting started

## For developers

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
If you do not want to change your system interpreter you can use `conda` to install a compatible Python version.
In this case, activate the respective environment before running the installation commands below. 

*gaitmap* uses [poetry](https://python-poetry.org) to manage its dependencies.
If you have trouble installing `poetry` while using `zsh` as your shell, check this [issue](https://github.com/python-poetry/poetry/issues/507)
Once you installed poetry, run the following commands to initialize a virtual env and install all development dependencies:

```bash
poetry install --no-root
```

This will create a new folder called `.venv` inside your project dir.
It contains the python interpreter and all site packages.
You can point your IDE to this folder to use this version of Python.
For PyCharm you can find information about this [here](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)

### Testing, linting, etc.

This library uses `pytest` for testing.
You can run it from the commandline (from the project root) using:

```bash
poetry run pytest
```
Alternatively, you can use the `pytest` integration of your IDE.

To ensure consistent code style the library uses strict linting rules.
You can check your code against these rules using `prospector`.

```bash
poetry run prospector
```

To make live easier for you, you should use [black](https://github.com/psf/black) to autoformat your code.
Just run the following from the commandline:

```bash
poetry run black .
```

Alternatively, you can integrate `black` [into you editor](https://black.readthedocs.io/en/stable/editor_integration.html).

### Configure IDE

#### PyCharm

- Set docstring convention to `numpy`
- Set default testrunner to `pytest`



