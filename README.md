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

To make it easier to run commandline tasks we use [doit](https://pydoit.org/contents.html) to provide a cross-platform 
cli for common tasks.
All commands need to be executed in the `venv` created by poetry.

To list the available tasks, run:

```bash
$ poetry run doit list
docs     Build the html docs using Sphinx.
format   Reformat all files using lint.
lint     Lint all files with Prospector.
test     Run Pytest with coverage.
```

To run one of the commands execute (e.g. the `test` command):
```bash
poetry run doit test
```

To execute `format`, `lint`, and `test` all together, run:
```bash
poetry run doit
# or if you want less output
petry run doit -v 0
```

Tou should run this as often as possible!
At least once before any `git push`.

**Protip**: If you do not want to type `poetry run` all the time, you can also activate the `venv` for your current
terminal session using `poetry shell`.
After this you can just type, for example, `doit test`.

#### Tools we are using

This library uses `pytest` for **testing**. Besides using the command above, you can also use an IDE integration available
for most IDEs.
For *PyCharm* you just need to set the default testrunner to `pytest`.

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into you editor](https://black.readthedocs.io/en/stable/editor_integration.html), if you
do not want to run it from the commandline.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we us *prospector* to handle all other **linting** tasks.
*Prospector* runs `pylint`, `pep257`, and `pyflakes` with custom rules to ensure consistent code and docstring style.

For **documentation** we follow the numpy doc-string guide lines and autobuild our API documentation using *Sphinx*.
To make your live easier, you should also set your IDE tools to support the numpy docstring conventions.
