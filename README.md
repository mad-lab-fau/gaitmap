# gaitmap - The Gait and Movement Analysis Package

*gaitmap* provides a set of algorithms to analyse your IMU movement data without getting into your way.
Its API is designed to mimic `sklearn` to provide you a familiar and elegant interface.


[![pipeline status](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/badges/master/pipeline.svg)](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commits/master)
[![coverage report](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/badges/master/coverage.svg)](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commits/master)
[![docs](https://img.shields.io/badge/docs-online-green.svg)](http://MadLab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/README.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Getting started

### Installation

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
Then install the package using pip.

For a stable experience install one of our releases (e.g. 1.2.0):

For available versions, see the [release page](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/releases).
```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap.git@v1.2.0 --upgrade
```

The latest git (bleeding edge) version:
```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap.git --upgrade
```

If you are planning to make any changes to the library, please refer to the developer section below.

### Working with Algorithms

*gaitmap* is designed to be a toolbox and not a single algorithm.
This means you are expected to pick and use individual algorithms.
A good way to get an overview over the available algorithms and possibilities is to look at the
[Examples](http://madlab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/auto_examples/index.html).
It is also highly advisable to read through the guides on
[Coordinate Systems](http://madlab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/source/user_guide/coordinate_systems.html)
and the
[Common Datatypes](http://madlab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/source/user_guide/datatypes.html).

## For Developers

The [Development Guide](http://madlab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/source/development/development_guide.html)
and the
[Project Structure Guide](http://madlab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/source/development/project_structure.html)
have detailed information for all new developers.
Below, we included some very basic information as a quick reference here in the README.

Install Python >3.7 and [poetry](https://python-poetry.org).
Then run the commands below to get the latest source and install the dependencies:

```bash
git clone https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap.git
poetry install
```

To run any of the tools required for the development workflow, use the doit commands:

```bash
$ poetry run doit list
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
register_ipykernel   Add a jupyter kernel with the gaitmap env to your local install.
test                 Run Pytest with coverage.
type_check           Type check with mypy.
```
