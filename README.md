<img src="./docs/_static/logo/gaitmap_logo_with_text.png" height="200">

# gaitmap - The Gait and Movement Analysis Package

*gaitmap* provides a set of algorithms to analyze your IMU movement data without getting into your way.
Its API is designed to mimic `sklearn` to provide you a familiar and elegant interface.


[![pipeline status](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/badges/master/pipeline.svg)](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commits/master)
[![coverage report](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/badges/master/coverage.svg)](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commits/master)
[![docs](https://img.shields.io/badge/docs-online-green.svg)](http://MadLab.mad-pages.informatik.uni-erlangen.de/GaitAnalysis/gaitmap/README.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Getting started

### Installation

*gaitmap* only supports Python 3.8 and newer.
First, install a compatible version of Python.

Then you need to install the provided packages.
Gaitmap is split into two packages: `gaitmap` and `gaitmap_mad`.
To get access to all available algorithms, you need to install both packages.

**For now, simply installing gaitmap will install both packages, but this will change in the future!**

For a stable experience install one of our releases (e.g. 2.0).

All new releases (>=2.0) can be found on [Github](https://github.com/mad-lab-fau/gaitmap/releases).
All old releases are available via MaD-Lab internal [Gitlab](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/releases).

```
# gaitmap
pip install git+https://github.com/mad-lab-fau/gaitmap.git@v2.0.0 --upgrade
```

The latest git (bleeding edge) version:
```
# gaitmap
pip install git+https://github.com/mad-lab-fau/gaitmap.git --upgrade
```

<!-- To install the package using poetry, make sure you use a version newer than 1.2.0b2.
This is the first version of poetry that supports subdirectories for git dependencies.
Note, that even then, there are a couple of bugs with poetry`s subdirectory support.
Hence, we would recommend to use the package versions of gaitmap and gaitmap_mad and not install them from source. -->

#### Enabling specific features

- Hidden Markov Models: To use the HMM based algorithms make sure you install `gaitmap` with the `hmm` extra.
  ```
  pip install "git+https://github.com/mad-lab-fau/gaitmap.git[hmm]" --upgrade
  ```
  and make sure that `gaitmap_mad` is installed.
  This installs the `pomegranate` package, which is the basis for the HMM implementation.
  Note, that we only support the `pomegranate` version `>=0.14.2,<=0.14.6` and that `pomegrante` is not compatible with 
  Python 3.10.

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

Install Python >=3.8 and [poetry](https://python-poetry.org).
Then run the commands below to get the latest source and install the dependencies:

```bash
git clone https://github.com/mad-lab-fau/gaitmap.git
poetry install
```

Note, that you don't need to care about the `gaitmap_mad` subpackage.
All dependencies are specified in the main `pyproject.toml` and the `gaitmap_mad` will be installed in editable mode
when running `poetry install`.

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
update_version       Bump the version in pyproject.toml and gaitmap.__init__ .
```
