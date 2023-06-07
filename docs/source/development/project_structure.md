# Scope and Code Structure

To ensure that the project is easy to use, easy to maintain, and easy to expand in the future, all developers should
adhere to the guidelines outlined below.
Further the developers should familiarize themselves with aim and the scope of the project to make better decision when
it comes to including new functionality.
If you are new developer and need to familiarize yourself and need help with setting up your development environment to
work on this project, have a look at the [Development Guide](development_guide.md).

**Everything that follows are recommendations.**

As for every project you should:
- value your future self over your current self (don't use shortcuts or hacks that might become liabilities in the
long-term).
- think before you act.
- know the rules, before you break them.
- ask if in doubt.
- ask for a third opinion if you have two competing ones.

## Aim and Scope

The aim of the project is to provide a **library** that provides all the necessary tools to run a state-of-the-art gait
analysis.
This means this library is meant to be a toolbox and specifically **not** a single pipeline.
In the long run, this library should include multiple algorithms that allow users to build a multitude of different
pipelines.

Following the reasoning, the library should not force users into a "certain way of doing thing".
This means specifically:

1. Classes and functions should only require the bare minimum of input they require
2. Variations in functionality should be made available to users either by providing appropriate keyword arguments,
   separate methods, or even separate functions or classes.
3. Following 2., each function/class should have one function and one function only.
   In the case where "one function" consist of multiple steps (e.g. event detection consists of HS-detection, 
   TO-detection, MS-detection) and it makes sense to group them together to provide an easier interface, the individual 
   functions should still be available to users to allow him to skip or modify steps as they desire.
4. The library should not prevent users from "intentional stupidity".
   For example, if users decide to apply an event detection method designed to only run on data from sensors attached to 
   the shoe on data from a hip sensor, **let them**.
5. Whenever possible the library should allow to provide native datatypes as outputs.
   Native datatypes in this case includes all the container objects Python supports (lists, dicts, etc.) and the base 
   datatypes of `numpy` and `pandas` (np.array, pd.DataFrame, pd.Series). Only if it improves usability and adds 
   significant value (either for understanding or manipulating the output), custom datatypes should be used.
   One example of that would be the `tpcp.Dataset` datatype used for high level pipelines.
6. The library should be agnostic to sensor systems and should not contain any code that is highly specific to a certain
   IMU system. This means that loading and preprocessing should be handled by the user or other libraries.
   However, gaitmap can provide utils to make preprocessing easier for users.
   (Loading is out of scope, as to many data formats exist and it is not feasible to support all of them.)


## Code Structure

### Library Structure

For the `gaitmap` vs. `gaitmap_mad` discussion, see [here](gaitmap_mad).

As the library aims to support multiple algorithms, each algorithm with similar function should be grouped into 
individual modules/folders (e.g. Stride-Segmentation, Event Detection, Orientation Estimation, ...).
Each algorithm should than be implemented in a separate file unless multiple algorithms are really tightly coupled 
e.g. the scaler methods: `gaitmap/data_transform/_scaler.py`.
If an algorithm requires large amount of code and multiple classes/functions, it can be refactored into its own
submodule (for example see `gaitmap_mad/gaitmap_mad/stride_segmentation/hmm`)

### Import Paths
To ensure that users can use short import path, we discourage the import from the python files directly.
Instead, new algorithms/functions should be added to the init of the parent folder to be accessible from there.
The implementation file itself, should have a leading underscore, to mark it as private and discourage users from 
importing from it.

For example, the algorithm `HerzerEventDetection` is defined in `gaitmap/event_detection/_herzer_event_detection.py`.
This means the recommended import should be `from gaitmap.event_detection import HerzerEventDetection`.
To make that possible, `HerzerEventDetection` is imported in `gaitmap/event_detection/__init__.py`.

In the respective `__init__` files, the `__all__` list should be used to allow for `*` imports without cluttering the 
namespace.

In general, we use absolute imports internally (instead of relative imports), even though it requires more typing.

.. warning:: With absolute imports and the use of `__init__` imports, you can sometimes create circular dependencies.
   E.g. when you want to import a function `a()` defined in `mod/_a.py` in `mod/_b.py` and `mod/_b.py` and `mod/_a.py`
   are both imported in `mod/__init__.py` (as by convention explained above), you must import `c()` with its full 
   internal path `from mod._c import c` instead of `from mod import c` to avoid circular imports.

In case a single function from a external package is used, just import this function.
In case multiple functions from an external package are used, import this package/module under a commonly used alias
(e.g. `np` for numpy, `pd` for pandas, ...)

### Algorithms with specific dependencies

Some algorithms require specific dependencies, that are ot required by anything else.
In this case, they should be made optional dependencies of gaitmap to reduce the number of dependencies for users.
To make this possible, the algorithm should be implemented in a separate submodule 
(e.g. `gaitmap_mad/gaitmap_mad/stride_segmentation/hmm`), that is not imported anywhere else in the library.

In case, where only a few methods from a different library are required, it might also be feasible to just copy the
required methods into the submodule and add a comment to the source of the code.
Make sure to check the license of the original code and provide sufficient attribution.
One example of this can be found in `gaitmap_mad/stride_segmentation/dtw/_vendored_tslearn.py`.

### Helper Functions and Utils

Functions that can be reused across multiple algorithms of similar type should be placed in a module level `_utils.py` 
file (e.g. `stride_segmentation/_utils.py`). Functions that are reusable across multiple modules should be placed in an
appropriate file in the package level `utils` module (e.g. `utils/_math_helper.py`).
Utils are generally "private", but when you can imagine usecases where users might reasonably need to use them when 
working with the library, you can expose them in the respective `__init__` files.

### Class Structure

All larger algorithms should be represented by classes and not by functions for a couple of reasons that are explained 
below.
See the [general guides in tpcp](https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html).

From the guide:

- The `__init__` of each class should only be there to set parameters. No parameter validation or any functionality 
  should be performed here.
- No actual data should be passed to the `__init__`. Think of it as configuring the algorithm.
- Defaults for **all** parameters should be provided in the `__init__`.
- The names of the class attributes should be identical to the parameter names used in the `__init__`
  (i.e. the init should contain statements like: `self.parameter_name = parameter_name`).
- All parameters that are not directly depended on the input data, should be set in the `__init__` and not in the
  *action* method (see below).
  This also includes parameters that should be adapted based on the data, but can theoretically estimated without having
  the data available (e.g. the optimal threshold for DTW).
  All these parameters should be set in the `__init__`.
  The data and all other measured/directly data-depended parameters are passed in the action method.
  This includes for example, the raw IMU data, the sampling rate demographic information, etc..
- Results and outputs are stored with a trailing underscore (e.g. `filtered_stride_list_`.
- All algorithms of the same type should have a consistent interface with (as far as possible), identical input 
  parameters to allow drop-in replacements of different algorithms
- Each type of algorithm has one (or multiple) "action" methods with a descriptive name.
  These *action* methods take the actual data as input and will produce results.
- All *action* methods just return `self` (the object itself)
- Multiple action methods might be required in one of the following cases:
    - Give users more granular functionality (e.g. a `detect_hs` method in addition to a general `detect` method)
    - Multiple steps are required for the algorithm (e.g. fit and predict for ML based algos)
    - Alternative functionality that can/should not be handled by arguments (e.g. `fit` vs `fit_proba` in sklearn)


Additions to the guide:

- All algorithms classes must directly or indirectly inherit from `tpcp.Algorithm`, and `gaitmap.base._BaseSerializable`
- All classes should store the data (and other arguments) passed in the "action" step in the class object unless the 
  amount of data would result in an unreasonable performance issue.
  Ideally this should be a reference and not a copy of the data! This allows to path the final object as a whole to 
  helper functions, that e.g. can visualize in and outputs.
  These parameters should be documented under "Other Parameters" to not clutter the docstring.
- Mutable defaults in the init are as always a bad idea, but in gaitmap we make specific exceptions.
  See the section on *Mutable defaults* below.
- All methods should take care that they do not modify the original data passed to the function.
  If required a copy of the data can be created, but **not** stored in the object.
- All classes should validate their input parameters during the "action" (or whenever the parameters are first needed).
  Don't overdue the validation and focus on logical validation (e.g. a value can not be larger than x) and not on type 
  validation.
  For type validation, we should trust that Python provides the correct error message once an invalid step is performed.
- All classes should inherent from a BaseClass specific to their type that implements common functionality and enforces 
  the interface. Remember to call respective `super` methods when required.
  The resulting class structure should look like this:
```
tpcp.Algorithm/gaitmap.base._BaseSerializable -> Basic setting of parameters
|
Base<AlgorithmType> -> Basic interface to ensure all algos of the same type use the same input and outputs for their 
|                      action methods
|
<TheActualAlgorithm>
|
<VariationsOfAlgorithm> -> A improved version of an algorithm, when it does not make sense to toggle the improvement 
                           via an inputparameter on the algorithm
```


#### Example class structure

Below you can find the simplified class structure of the `RamppEventDetection` algorithm.
It should serve as an example on how further algorithms should be implemented and documented.
Also review the actual implementation of the other algorithms for further inspiration and guidance.

```python
import numpy as np
import pandas as pd
from gaitmap.base import BaseEventDetection, BaseType
from typing import Optional, Tuple, Union, Dict
from gaitmap.utils.datatype_helper import SensorData, is_multi_sensor_data, is_single_sensor_data


class RamppEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    RamppEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the signals as described in Rampp et al. (2014) [1]_.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    ic_search_region_ms
        ...
    min_vel_search_win_size_ms
        ...

    Attributes
    ----------
    min_vel_event_list_ : A stride list or dictionary with such values
        ...
    start_ : 1D array or dictionary with such values
        ...
    end_ : 1D array or dictionary with such values
        ...
    tc_ : 1D array or dictionary with such values
        ...
    min_vel_ : 1D array or dictionary with such values
        ...
    ic_ : 1D array or dictionary with such values
        ...
    pre_ic_ : 1D array or dictionary with such values
        ...

    Other Parameters
    ----------------
    data
        ...
    sampling_rate_hz
        ...
    stride_list
        ...

    """

    ic_search_region_ms: Tuple[float, float]
    min_vel_search_win_size_ms: float

    start_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    end_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    tc_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    min_vel_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    ic_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    pre_ic_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    stride_events_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: SensorData
    sampling_rate_hz: float
    segmented_stride_list: pd.DataFrame

    def __init__(self, ic_search_region_ms: Tuple[float, float] = (80, 50), min_vel_search_win_size_ms: float = 100):
        # Just add the Parameters without any logic
        self.ic_search_region_ms = ic_search_region_ms
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms
        super().__init__()

    def detect(self: BaseType, data: SensorData, sampling_rate_hz: float,
               segmented_stride_list: pd.DataFrame) -> BaseType:
        """Find gait events in data within strides provided by stride_list.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        sampling_rate_hz
            The sampling rate of the data
        segmented_stride_list
            A list of strides provided by a stride segmentation method

        Returns
        -------
        self
            The class instance with all result attributes populated

        Examples
        --------
        Get gait events from single sensor signal

        >>> event_detection = RamppEventDetection()
        >>> event_detection.detect(data=data, sampling_rate_hz=204.8, stride_list=stride_list)
        >>> event_detection.min_vel_event_list_
            s_id   start     end      ic      tc  min_vel  pre_ic
        0      0   519.0   710.0   651.0   584.0    519.0   498.0
        1      1   710.0   935.0   839.0   802.0    710.0   651.0
        2      2   935.0  1183.0  1089.0  1023.0    935.0   839.0
        ...

        """
        # Set all "Other Parameters"
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.segmented_stride_list = segmented_stride_list

        # Potential validation of parameters should be performed here

        if is_single_sensor_data(data):
            # Handle single sensors
            ...
        elif is_multi_sensor_data(data):
            # Handle multiple sensors
            ...
        else:
            raise ValueError("Provided data set is not supported by gaitmap")

        return self
```

### Random and Initial State

If any algorithms rely on random processes/operations, the random state should be configurable, by an optional kwarg in
the `__init__` called `random_state`.
We follow the [`sklearn` recommendations](https://scikit-learn.org/stable/glossary.html#term-random-state) on this.

Algorithms that require an initial value for some optimization should expose this value via the `__init__`.
If the parameter is `None` a random initial value should be used that is controlled by the additional `random_state`
argument.

### Mutable Defaults

See [this guide in tpcp](https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html#mutable-defaults).


## Code guidelines

All code should follow coherent best practices.
As far as possible the adherence to these best practices should be tested using linters or a testsuite that runs as part
of the CI.

### General Codestyle

For general codestyle we follow [PEP8](https://www.python.org/dev/peps/pep-0008/) with a couple of exceptions
(e.g. line length).
These are documented in the linter config (`.prospector.yml`)

### Naming

We follow the naming conventions outlined in [PEP8](https://www.python.org/dev/peps/pep-0008/#naming-conventions).

For algorithms (if no better name is available) we use `AuthorNameType` (e.g. `BarthEventDetection`).

### Documentation

For documentation, we follow [numpys guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
If the datatype is already provided as TypeHint (see below) it does not need to be specified in the docstring again.
However, it might be helpful to document additional type information (e.g. the shape of an array that can not be
captured by the TypeHint)

All user-facing functions (all functions and methods that do not have a leading underscore) are expected to be properly
and fully documented for potential users.
All private functions are expected to be documented in a way that other developer can understand them.
Additionally, each module should have a docstring explaining its content.
If a module contains only one class this can a single sentence/word (e.g. `"""Event detection based on ... ."""`).

### Typehints

To provide a better developer experience the library should use
[TypeHints](https://numpydoc.readthedocs.io/en/latest/format.html) where ever possible.

Remember to use `np.ndarray` instead of `np.array` as type specification of numpy arrays.
