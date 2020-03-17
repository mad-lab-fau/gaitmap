# gaitmap - The Gait and Movement Analysis Package

*gaitmap* provides a set of algorithms to analyse your IMU movement data without getting into your way.
It's API is designed to mimic `sklearn` to provide you an familiar and elegant interface

## Getting started

## For developers

*gaitmap* only supports Python 3.7 and newer.
First, install a compatible version of Python.
If you do not want to change your system interpreter you can use `conda` to install a compatible Python version.
In this case, activate the respective environment before running the installation commands below. 

*gaitmap* uses [poetry](https://python-poetry.org) to manage its dependencies.
Once you installed poetry, run the following commands to initialize a virtual env and install all development dependencies:

```bash
poetry install --no-root
```

This will create a new folder called `.venv` inside your project dir.
It contains the python interpreter and all site packages.
You can point your IDE to this folder to use this version of Python.
For PyCharm you can find information about this [here](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)



