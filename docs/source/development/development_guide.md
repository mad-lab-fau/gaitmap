# Development Guide

This document contains information for developers that need further in-depth information on how to setup and use tools
and learn about programing methods used in development of this project.

If you are looking for a higher level overview over the guiding ideas and structure of this project, please visit the
[Project Structure document](project_structure.md).

## Project Setup and Poetry

*gaitmap* only supports Python 3.8 and newer.
First, install a compatible version of Python.
If you do not want to modify your system installation of Python you can use [conda](https://docs.conda.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv).
However, there are some issues with using conda.
Please, check the [trouble shooting guide](#trouble-shooting) below.

*gaitmap* uses [poetry](https://python-poetry.org) to manage its dependencies.
First install poetry `>=1.2`.
Once you installed poetry, run the following commands to initialize a virtual env and install all development
dependencies:

```bash
poetry env use "path/to/python/you/want/to/use"
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

To make it easier to run commandline tasks we use [poethepoet](https://github.com/nat-n/poethepoet) to provide a 
cross-platform cli for common tasks.
All commands need to be executed in the `venv` created by poetry.

To list the available tasks, run:

```bash
$ poetry run poe
...
CONFIGURED TASKS
  format            
  lint              Lint all files with Prospector.
  check             Check all potential format and linting issues.
  test              Run Pytest with coverage.
  docs              Build the html docs using Sphinx.
  register_jupyter  Register the gaitmap environment as a Jupyter kernel for testing.
  version           Bump version in all relevant places.
  bump_dev          Update all dev dependencies to their @latest version.

```

To run one of the commands execute (e.g. the `test` command):
```bash
poetry run poe test
```

**Protip**: If you installed poethepoet globally, you can skip the `poetry run` part at the beginning.

### Formatting and Linting

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into you editor](https://black.readthedocs.io/en/stable/editor_integration.html), if you
do not want to run it from the commandline.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we us *ruff* to handle all other **linting** tasks.

For **documentation** we follow the numpy doc-string guide lines and autobuild our API documentation using *Sphinx*.
To make your live easier, you should also set your IDE tools to support the numpy docstring conventions.

To run formatting you can use

```bash
poetry run poe format
```

and for linting you can run

```bash
poetry run poe lint
```

Tou should run this as often as possible!
At least once before any `git push`.


### Testing and Test data

This library uses `pytest` for **testing**. Besides using the poe-command, you can also use an IDE integration
available for most IDEs.
From the general structure, each file has a corresponding `test_...` file within a similar sub structure.

#### Common Tests

For basically all new algorithms we want to test a set of basic functionalities.
For this we have `tests.mixins.test_algorithm_mixin.TestAlgorithmMixin` (There is also a caching mixin 
`TestCachingMixin` for algorithms that allow a `memory` parameter).
To use the general mixin, create a new test class, specify the `algorithm_class`, the `after_action_instance` fixture
and set `__test__ = True`.

Below an example from `PcaAlignment`:

```python
import pytest

from gaitmap.preprocessing.sensor_alignment import PcaAlignment
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

class TestMetaFunctionality(TestAlgorithmMixin):
    __test__ = True
    
    algorithm_class = PcaAlignment

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> PcaAlignment:
        pcaa = PcaAlignment()
        pcaa.align(healthy_example_imu_data["left_sensor"].iloc[:10])
        return pcaa
```

This will test basic things like cloning, and the `get_params` and `set_params` methods.

#### Test Data

Test data is available in the `example_data` folder.
Within scripts or examples, the recommended way to access it is using the functions in `gaitmap.example_data`.

```python
from gaitmap.example_data import get_healthy_example_imu_data

data = get_healthy_example_imu_data()
```

Within tests you can also use the pytest fixtures defined `tests/conftest.py`.

```python
# Without import in any valid test file

def test_myfunc(healthy_example_imu_data):
    ...
```

#### Testing Examples

For each mature feature their should also be a corresponding example in the `examples` folder.
To make sure they work as expected, we also test them using `pytest`.
For this create a new test function in `tests/test_examples/test_all_examples.py` and simply import the example
within the respective function.
This will execute the example and gives you access to the variables defined in the example.
They can then be tested.
Most of the time a regression/snapshot test is sufficient (see below).

#### Snapshot Testing


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

#### Manual Testing

While all automated tests should go in the test folder, it might be helpful to create some external test script from 
time to time.
For this you can simply install the package locally (using `poetry install`) and even get a Jupyter kernel with all
dependencies installed (see [IDE Config](#Configure-your-IDE)).

 
## Configure your IDE
(Configure-your-IDE)=

### Pycharm

**Test runner**: Set the default testrunner to `pytest`. 

**Black**: Refer to this [guide](https://black.readthedocs.io/en/stable/editor_integration.html) 

**Autoreload for the Python console**:

You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to
settings->Build,Excecution,Deployment->Console->Python Console in the Starting Script:

```python
%load_ext autoreload
%autoreload 2
```

### Jupyter Lab/Notebooks

While we do not (and will not) use Jupyter Notebooks in gaitmap, it might still be helpful to use Jupyter to debug and
prototype your scientific code.
To set up a Jupyter environment that has gaitmap and all dependencies installed, run the following commands:

```
# poetry install including root!
poetry install
poetry run poe register_ipykernel
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

As long as no new minor or major version is released, all changes should be interface compatible.
This means that the user can update to a new patch version without changing any user code!

This means at any given time we need to support and work with two versions:
The last minor release, which will get further patch releases until its end of life.
The upcoming minor release for which new features are developed at the moment.
However, in most cases we will also not create proper patch releases, but expect users to update to the newest git 
version, unless it was an important and major bug that got fixed.

Note that we will not support old minor releases after the release of the next minor release to keep things simple.
We expect users to update to the new minor release, if they want to get new features and bugfixes.

To make such a update model go smoothly for all users, we keep an active changelog, that should be modified a feature
is merged or a bug fixed.
In particular changes that require updates to feature code should be prominently highlighted in the "Migration Guide"
section.

There is no fixed timeline for a release, but rather a list of features we will plan to include in every release.
Releases can happen often and even with small added features.


## Git Workflow

As multiple people are expected to work on the project at the same time, we need a proper git workflow to prevent issues.

### Branching structure

This project uses (as of version 1.2.0) a master + feature branches.
This workflow is well explained [here](https://www.atlassian.com/blog/git/simple-git-workflow-is-simple).
  
All changes to the master branch should be performed using feature branches.
Before merging, the feature branches should be rebased onto the current master.

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

# Create a merge request and merge via web interface

# Once branch is merged, delete it locally, start a new branch
git checkout master
git branch -D new-branch-name

# Start at top!
```

### For large features

When implementing large features it sometimes makes sense to split it into individual merge requests/sub-features.
If each of these features are useful on their own, they should be merged directly into master.
If the large feature requires multiple merge requests to be usable, it might make sense to create a long-lived feature
branch, from which new branches for the sub-features can be created.
It will act as a develop branch for just this feature.
Remember, to rebase this temporary dev branch onto master from time to time.

.. note:: Due to the way gaitmap is build, it is often possible to develop new features (e.g. algorithms) without
          touching the gaitmap source code.
          Hence, it is recommended to devlop large features in a separate repository and only merge them into gaitmap
          once you worked out all the kinks.
          This avoids long living feature branches in gaitmap and allows you to develop your feature in a more flexible 
          way.

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
(trouble-shooting)=


### `poetry not found` when using `zsh` as shell

If you have trouble installing `poetry` while using `zsh` as your shell, check this [issue](https://github.com/python-poetry/poetry/issues/507)

### Installation issues while using `conda`

.. note:: This might be outdated! If you run into any issues, please check google :)

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
    
### Warning/Error about outdated/missing dependencies in the lock file when running `install` or `update`

This happens when the `pyproject.toml` file was changed either by a git update or by manual editing.
To resolve this issue, run the following and then rerun the command you wanted to run:

```bash
poetry update --lock
``` 

This will synchronise the lock file with the packages listed in `pyproject.toml` 
