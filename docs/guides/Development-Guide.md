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

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into you editor](https://black.readthedocs.io/en/stable/editor_integration.html), if you
do not want to run it from the commandline.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we us *prospector* to handle all other **linting** tasks.
*Prospector* runs `pylint`, `pep257`, and `pyflakes` with custom rules to ensure consistent code and docstring style.

For **documentation** we follow the numpy doc-string guide lines and autobuild our API documentation using *Sphinx*.
To make your live easier, you should also set your IDE tools to support the numpy docstring conventions.


## Testing and Test data

This library uses `pytest` for **testing**. Besides using the doit-command, you can also use an IDE integration
available for most IDEs.

While all automated tests should go in the test folder, it might be helpful to create some external test script form 
time to time.
For this you can simply install the package locally (using `poetry install`) and even get a Jupyter kernel with all
dependencies installed (see [IDE Config](#Configure-your-IDE)).
Test data is available under `example_data` and you can import it directly using the `get_...` helper functions in 
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
 
## Configure your IDE


#### Pycharm

**Test runner**: Set the default testrunner to `pytest`. 

**Black**: Refer to this [guide](https://black.readthedocs.io/en/stable/editor_integration.html) 

**Autoreload for the Python console**:

You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to
settings->Build,Excecution,Deployment->Console->Python Console in the Starting Script:

```python
%load_ext autoreload
%autoreload 2
```


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

## Release Model

Gaitmap follows typically semantic visioning: A.B.C (e.g. 1.3.5)

- `A` is the major version, which will be updated once there were fundamental changes to the project
- `B` is the minor version, which will be updated whenever new features are added
- `C` is the patch version, which will be updated for bugfixes

As long as no new minor or major version is released, all changes must be interface compatible.
This means that the user can update to a new patch version without changing any user code!

This means at any given time we need to support and work with two versions:
The last minor release, which will get further patch releases until its end of life.
The upcoming minor release for which new features are developed at the moment.

Note that we will not support old minor releases after the release of the next minor release to keep things simple.
We expect users to update to the new minor release, if they want to get new features and bugfixes.
In some rare cases we might consider backporting certain bug fixes/small features.

To make such a update model go smoothly for all users, we keep an active changelog, that should be modified a feature
is merged or a bug fixed.
In particular changes that require updates to feature code should be prominently highlighted in the "Migration Guide"
section.

There is no fixed timeline for a release, but rather a list of features we will plan to include in every release.
Releases can happen often and even with small added features.

In most cases we will also not create proper patch releases, but expect users to update to the newest git version,
unless it was an important and major bug that got fixed.


## Git Workflow

As multiple people are expected to work on the project at the same time, we need a proper git workflow to prevent issues.

### Branching structure

As we need to support two versions at any given time (see previous section), we need a model with at least two main
branches: master and develop.

- The master branch is there to support the current release.
  This means all bugfixes and patches will be committed here
- The develop branch is there to plan and create the next release.
  All new features should be committed here.
  The develop branch will get all patch releases committed to master either by cherry-picking or by rebase.
  As rebasing a shared branch is "dangerous" all developers should be informed when this happens (see more below).
  
All changes to these two main branches should be performed using feature branches.

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
# Create a new branch (master for bug fixes, develop for features)
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

### Rebasing Develop

If develop is rebased onto master and feature branches exist, that were started in an old version of develop, the
following can happen:

```
master ---A---B------------------------C
           \                            \
            D---E---F develop(old)       D---E---F develop
                     \
                      G---H feature
```

To solve this situation, the developer of `feature` should take the following steps:

```
git checkout develop  # You current version
git pull origin develop  # Get the most up to date version of develop
git checkout feature
git log  # Find the last commit before you started you feature (the last commit on develop(old)) and copy it
git rebase --onto develop <commit-hash> feature
git push origin feature --force-with-lease
```

This will not rebase D-G onto develop, but only the commits between `<commit-hash>` and `feature` (aka G and H).
For more info you can also check this
[stack overflow post](https://stackoverflow.com/questions/31881885/how-to-rebase-a-branch-off-a-rebased-branch)

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
