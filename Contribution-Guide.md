To ensure that the project is easy to use, easy to maintain, and easy to expand in the future, all developers should adhere to the guidelines outlined below.
Further the developers should familiarize themselves with aim and the scope of the project to make better decision when it comes to including new functionality.

**Everything that follows are recommendations.**

As for every project you should:
- value your future self over your current self (don't use shortcuts or hacks that might become liabilities in the long-term).
- think before you act.
- know the rules, before you break them.
- ask if in doubt.
- ask for a third opinion if you have two competing ones.

## Aim and Scope

The aim of the project is to provide a **library** that provides all the necessary tools to run a state-of-the-art gait analysis.
This means this library is meant to be a toolbox and specifically **not** a single pipeline.
In the long run, this library should include multiple algorithms that allow the user to build a multitude of different pipelines.

Following the reasoning, the library should not force the user into a "certain way of doing thing".
This means specifically:

1. Classes and functions should only require the bare minimum of input they require
2. Variations in functionality should be made available to the user either by providing appropriate keyword arguments, separate methods, or even separate functions or classes.
3. Following 2., each function/class should have one function and one function only.
In the case where "one function" consist of multiple steps (e.g. event detection consists of HS-detection, TO-detection, MS-detection) and it makes sense to group them together to provide a easier interface, the individual functions should still be available to the user to allow him to skip or modify steps as he desires.
4. The library should not prevent the user from "intentional stupidity".
For example, if the user decides to apply a event detection method designed to only run on data from sensors attached to the shoe on data from a hip sensor, **let him**.
5. Whenever possible the library should allow to provide native datatypes as outputs.
Native datatypes in this case includes all the container objects Python supports (lists, dicts, etc.) and the base datatypes of `numpy` and `pandas` (np.array, pd.DataFrame, pd.Series). Only if it improves usability and adds significant value (either for understanding or manipulating the output), custom datatypes should be used. One example of this would be a "Stride" dataype, as this would help to keep all relevant information together in one place
6. Following 5, if custom datatypes are used, functionality needs to exist to convert them into native types that contain all the information.
Ideally, it should be possible to perform a conversion in both ways.
7. The library should be agnostic to sensor systems and should not contain any code that is highly specific to a certain IMU system. This means that loading and preprocessing should be handled by the user or other libraries.


## Code Structure

### Library Structure

As the library aims to support multiple algorithms, each algorithms with similar function should be grouped into individual modules/folders (e.g. Stride-Segmentation, Event Detection, Orientation Estimation, ...). Each algorithm should than be implemented in a separate file. If an algorithm requires large amount of code and multiple classes/functions, it can be refactored into its own submodule.

### Helper Functions and Utils

Functions that can be reused across multiple algorithms of similar type should be placed in a module level `utils.py` file (e.g. `stride_segmentation/utils.py`). Functions that are reusable across multiple modules should be placed in an appropriate file in the package level `utils` module (e.g. `modules/math_helper.py`)

### Class Structure

All larger algorithms should be represented by classes and not by functions for a couple of reasons that are explained below.
Further all main classes should adhere to the following structure.

This structure is heavily inspired by the interface of `sklearn` and will follow the developer guide outlined [here](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects) for the most part. Below the important points and differences are summarized. It is still recommended to read through the guide!

From the guide:

- The `__init__` of each class should only be there to set parameters. No parameter validation or any functionality should be performed here.
- No actual data should be passed to the `__init__`. Think of it as configuring the algorithm.
- Defaults for parameters should be provided in the `__init__`
- The names of the class attributes should be identical with the names used in the `__init__`.
- All parameters that are not directly depended on the input data, should be set in the `__init__` and not in the *action* method (see below). This also includes parameters that should be adapted based on the data, but can theoretically estimated without having the data available (e.g. the optimal threshold for DTW). All these parameters should be set in the `__init__`.
- Results and outputs are stored with a trailing underscore (e.g. `filtered_stride_list_`.
- All algorithms of the same type should have a consistent interface with (as far as possible), identical input parameters to allow drop-in replacements of different algorithms
- Each type of algorithm has one (or multiple) "action" methods with a descriptive name. These *action* methods take the actual data as input and will produce results.
- All *action* methods just return `self` (the object itself)
- Multiple action methods might be required in one of the following cases:
    - Give the user more granular functionality (e.g. a `detect_hs` method in addition to a general `detect` method)
    - Multiple steps are required for the algorithm (e.g. fit and predict for ML based algos)
    - Alternative functionality that can/should not be handled by arguments (e.g. `fit` vs `fit_proba` in sklearn)


Additions to the guide:

- All classes should store the data (and other arguments) passed in the "action" step in the class object unless the amount of data would result in an unreasonable performance issue. Ideally this should be a reference and not a copy of the data! This allows to path the final object as a whole to helper functions, that e.g. can visualize in and outputs.
These parameters should be documented under "Other Parameters" to not clutter the docstring.
- All methods should take care that they do not modify the original data passed to the function. If required a copy of the data can be created, but **not** stored in the object.
- All classes should have a `_validate(self)` method that handles validation of all the parameters. This function should be called during the "action" (or whenever the parameters are first needed).
Don't overdue the validation and focus on logical validation (e.g. a value can not be larger than x) and not on type validation. For type validation, we should trust that Python provides the correct error message once an invalid step is performed.
- All classes should inherent from a BaseClass specific to their type that implements common functionality and enforces the interface. Remember to call respective `super` methods when required.


#### Example class structure

```python

class AuthorNameEventDetection(BaseEventDetection):
    """ This is an example class.

    And here you would explain the full functionality.

    Parameters
    ----------
    parameter_one
        This is everything I know about parameter one
    parameter_two
        All the important stuff
    window_length : float, should be larger 0 and uneven
        Some info with extra type info

    Other Parameters
    ----------------
    data
        The data used for the action event

    Attributes
    ----------
    hs_events_
        The awesome output 1
    to_events_
        The awesome output 2
    
    Notes
    -----
    And here you would explain the algorithm in detail with references [1]

    .. [1] Author something something, Awesome paper

    Examples
    --------
    And of course some nice usage examples

    >>> awesome_stuff = EventDetection(parameter_one=5)

    """
    parameter_one: int
    parameter_two: str
    window_length: float


    data: np.ndarray

    hs_events_: np.ndarray
    to_events_: np.ndarray

    def __init__(self, parameter_one: int = 1, parameter_two: Optional[str] = None, window_length: float = 4.):
        self.parameter_one = parameter_one
        self.parameter_two = parameter_two
        self.window_length = window_length
        super().__init__()

    def detect(self: Type[T], data: np.ndarray) -> T:
        """This is the actual action method.
        
        Parameters
        ----------
        data:
            Your awesome input data

        Returns
        -------
        self
        """
        # Call super of base parent class
        # This should take care of potential preprocessing required.
        # In most cases this should call `_validate` implicit
        super().detect(data=data)

        self.data = data
        # Do some calculations
        self.hs_events_ = ...
        self.to_events_ = ...

        return self

    def detect_hs(self: Type[T], data: np.ndarray) -> T:
        """Example of secondary action function.
            
        This allows the user to have full control of what he/she 
        wants to do.

        Parameters
        ----------
        data:
            Your awesome input data

        Returns
        -------
        self
        """
        # Call validate manually
        self._validate()
        self.data = data
        # Do some calculations
        self.hs_events_ = ...

        return self

    def _validate(self) -> None:
        """Validate all parameters here."""

        if parameter_two is None:
            raise ValueError("`parameter_two` should a real value")
```

### Random and Initial State

If any algorithms rely on random processes/operations, the random state should be configurable, by a optional kwarg in the `__init__` called `random_state`. We follow the [`sklearn` recommendations](https://scikit-learn.org/stable/glossary.html#term-random-state) on this.

Algorithms that require an initial value for some optimization should expose this value via the `__init__`.
If the parameter is `None` a random initial value should be used that is controlled by the additional `random_state` argument.

## Code guidelines

All code should follow coherent best practices.
As far as possible the adherence to these best practices should be tested using linters or a testsuite that runs as part of the CI.

For a set of more general best practices when in comes to scientific code have a look at our [internal code guidelines](https://mad-srv.informatik.uni-erlangen.de/MaD-Public/mad-coding-guidelines)

### General Codestyle

For general codestyle we follow [PEP8](https://www.python.org/dev/peps/pep-0008/) with a couple of exceptions (e.g. line length).
These are documented in the linter config

### Naming

We follow the naming conventions outlined in [PEP8](https://www.python.org/dev/peps/pep-0008/#naming-conventions).

For algorithms (if not better name is available) we use `AuthorNameType` (e.g. `BarthEventDetection`).

### Documentation

For documentation we follow [numpys guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
If the datatype is already provided as TypeHint (see below) it does not need to be specified in the docstring again.
However, it might be helpful to document additional type information (e.g. the shape of an array that can not be captured by the TypeHint)

All user-facing functions (all functions and methods that do not have a leading underscore) are expected to be properly and fully documented for potential users.
All private functions are expected to be documented in a way that other developer can understand them.
Additionally each module should have a docstring explaining its content.
If a module contains only one class this can a single sentence/word (e.g. `"""Event detection based on ... ."""`).


### Typehints

To provide a better developer experience the library should use [TypeHints](https://numpydoc.readthedocs.io/en/latest/format.html) where ever possible.

Remember to use `np.ndarray` instead of `np.array` as type specification of numpy arrays.

### Imports

In case a single function from a external package is used, just import this function.
In case multiple functions from an external package are used, import this package/module under a commonly used alias (e.g. `np` for numpy, `pd` for pandas, ...)

For all package internal imports, use absolut imports.
