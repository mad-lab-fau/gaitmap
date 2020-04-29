# gaitmap - The Gait and Movement Analysis Package

*gaitmap* provides a set of algorithms to analyse your IMU movement data without getting into your way.
It's API is designed to mimic `sklearn` to provide you an familiar and elegant interface


[![pipeline status](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/badges/master/pipeline.svg)](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/commits/master)
[![coverage report](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/badges/master/coverage.svg)](https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/commits/master)
[![docs](https://img.shields.io/badge/docs-online-green.svg)](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/README.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Getting started

### Installation

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
Then install the package using pip:

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap.git --upgrade
```

Or manually with git (note that an editable install with `-e` is not possible for this project):

```
git clone https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap.git
cd gaitmap
pip install . --upgrade
```

If you are planning to make any changes to the library, please refer to the developer section below.

### Working with Algorithms

*gaitmap* is designed to be a toolbox and not a single algorithm.
This means you are expected to pick and use individually algorithms.
A good way to get an overview over the available algorithms and possibilities is to look at the
[Examples](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/auto_examples/index.html).
It is also highly advisable to read through the guides on
[Coordinate Systems](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Coordinate-Systems.html)
and the
[Default Datatypes](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Gaitmap-Datatypes.html).

## For developers

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
If you are using `conda` to manage your Python installation please refer to the 
[Developer Guide](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Development-Guide.html). 

*gaitmap* uses [poetry](https://python-poetry.org) to manage its dependencies during development.
Once you installed poetry, run the following commands to initialize a virtual env and install all development 
dependencies:

```bash
poetry install
```

In case you encounter any error and need more detailed instruction visit the 
[Developer Guide](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Development-Guide.html) and the 
[Project Structure Guide](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Project-Structure.html).

### Testing, linting, etc.

To make it easier to run commandline tasks we use [doit](https://pydoit.org/contents.html) to provide a cross-platform 
cli for common tasks.
All commands need to be executed in the `venv` created by poetry.

To list the available tasks, run:

```bash
$ poetry run doit list
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
register_ipykernel   Add a jupyter kernel with the gaitmap env to your local install.
test                 Run Pytest with coverage.
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

For any further information refer to the 
[Developer Guide](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Development-Guide.html).
