# Development Guide

This document contains information for developers that need further in-depth information on how to setup and use tools
and learn about programing methods used in development of this project.

If you are looking for a higher level overview over the guiding ideas and structure of this project, please visit the
[Project Structure document](Project-Structure.md).

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
poetry install
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

## Tools we are using

We are using [doit](https://pydoit.org/contents.html) to help you run all the tools.
See the [README](http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/README.html) for details.
Here is some further info about the tools that would be called by these commands:

This library uses `pytest` for **testing**. Besides using the doit-command, you can also use an IDE integration
available for most IDEs.
For *PyCharm* you just need to set the default testrunner to `pytest`.

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into your editor](https://black.readthedocs.io/en/stable/editor_integration.html), if you
do not want to run it from the commandline.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we us *prospector* to handle all other **linting** tasks.
*Prospector* runs `pylint`, `pep257`, and `pyflakes` with custom rules to ensure consistent code and docstring style.

For **documentation** we follow the [numpy doc-string guidelines](https://numpydoc.readthedocs.io/en/latest/format.html) and autobuild our API documentation using *Sphinx*.
To make your life easier, you should also set your IDE tools to support the numpy docstring conventions.

### Configure your IDE

#### Jupyter Lab/Notebooks

While we do not (and will not) use Jupyter Notebooks in gaitmap, it might still be helpful to use Jupyter to debug and
prototype your scientific code.
To set up a Jupyter environment that has gaitmap and all dependencies installed, run the following commands:

```
# poetry install including root!
poetry install
poetry run doit register_ipykernel
``` 

After this you can start Jupyter as always, but select "gaitmap" as a kernel when you want to run a notebook.

Remember to use the autoreload extension to make sure that Jupyter reloads gaitmap, when ever you change something in 
the library.
Put this in your first cell of every Jupyter Notebook to activate it:

```python
%load_ext autoreload  # Load the extension
%autoreload 2  # Autoreload all modules
```

#### Pycharm

You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to
settings->Build,Excecution,Deployment->Console->Python Console in the Starting Script:
```python
%load_ext autoreload
%autoreload 2
```

## Testing and Test data

While all automated tests should go in the test folder, it might be helpful to create some external test script form 
time to time.
For this you can simply install the package locally (using `poetry install`) and even get a Jupyter kernel with all
dependencies installed (see [IDE Config](#Configure-your-IDE)).
Test data is available under `test\example_data` and you can import it directly using the `get_...` helper functions in 
conftest:

```python
from tests.conftest import get_healthy_example_imu_data

data = get_healthy_example_imu_data()
```

If you can not import the tests folder, add the path to the gaitmap project folder (`gaitmap/`, **not** the package 
folder `gaitmap/gaitmap`) to your path at the top of your file:

```
import sys
sys.path.insert(0, "<path to the gaitmap project folder>")
```

The path can be relative to your current working directory.

### Regression Tests

To prevent unintentional changes to the data, this project makes use of regression tests.
These tests store the output of a function and compare the output of the same function at a later time to the stored
information.
This helps to ensure that a change did not modify a function unintentionally.
To make this easy, this library contains a small PyTest helper to perform regression tests.

A simple regression test looks like this:

```python
import pandas as pd

def test_regression(snapshot):
    # Do my tests
    result_dataframe = pd.DataFrame(...)
    snapshot.assert_match(result_dataframe)
```

This test will store `result_dataframe` in a json file if the test is run for the first time.
At a later time, the dataframe is loaded from this file to compare it.
If the new `result_dataframe` is different from the file content the test fails.

In case the test fails, the results need to be manually reviewed.
If the changes were intentionally, the stored data can be updated by either deleting, the old file
and rerunning the test, or by running ` pytest --snapshot-update`. Be careful, this will update all snapshots.

The results of a snapshot test should be committed to the repo.
Make reasonable decisions when it comes to the datasize of this data.

For more information see `tests/_regression_utils.py` or
`tests.test_stride_segmentation.test_barth_dtw.TestRegressionOnRealData.test_real_data_both_feed_regression` for an
 example.


## Git Workflow

As multiple people are expected to work on the project at the same time, we need a proper git workflow to prevent issues.

### Branching structure

For the initial development phase, we will use `master` + feature branchs. This is explained 
[here](https://guides.github.com/introduction/flow/index.html)

Remember, Feature branchs...:

- should be short-lived
- should be dedicated to a single feature
- should be worked on by a single person
- must be merged via a Merge Request and not manually
- must be reviewed before merging
- must pass the pipeline checks before merging
- should be rebased onto master if possible (remember only rebase if you are the only person working on this branch!)
- should be pushed soon and often to allow everyone to see what you are working on
- should be associated with a merge request, which is used for discussions and code review.
- that are not ready to review, should have a merge request prefixed with `WIP: `
- should also close issues that they solve, once they are merged

Workflow

```bash
# Create a new branch
git checkout master
git pull origin master
git checkout -b new-branch-name
git push origin new-branch-name
# Go to Gitlab and create a new Merge Request with WIP prefix

# Do your work
git push origin new-branch-name

# In case there are important changes in master, rebase
git fetch origin master
git rebase origin/master
# resolve potential conflicts
git push origin new-branch-name --force-with-lease

# If rebase is not feasible, merge
git fetch origin master
git merge origin/master
# resolve potential conflicts
git push origin new-branch-name

# Once branch is merged, delete it locally, start a new branch
git checkout master
git branch -D new-branch-name

# Start at top!
```

### General Git Tips

- Communicate with your Co-developers
- Commit often
- Commit in logical chunks
- Don't commit temp files
- Write at least somewhat [proper messages](https://chris.beams.io/posts/git-commit/)
   - Use the imperative mood in the subject line
   - Use the body to explain what and why vs. how
   - ...more see link above

## Trouble Shooting

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
