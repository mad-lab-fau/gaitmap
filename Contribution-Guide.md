To ensure that the project is easy to use, easy to maintain, and easy to expand in the future, all developers should adhere to the guidelines outlined below.
Further the developers should familiarize themselves with aim and the scope of the project to make better decision when it comes to including new functionality.

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

As the library aims to support multiple algorithms, each algorithms with similar function should be grouped into individual modules/folders (e.g. Stride-Segmentation, Event Detection, Orientation Estimation, ...). Each algorithm should than be implemented in a separate file. If an algorithms requires large amount of code and multiple classes/functions, it can be refactored into its own submodule.

### Helper Functions and Utils

Functions that can be reused across multiple algorithms of similar type should be placed in a module level `utils.py` file (e.g. `stride_segmentation/utils.py`). Functions that are reusable across multiple modules should be placed in an appropriate file in the package level `utils` module (e.g. `modules/math_helper.py`)

### Class Structure

All larger algorithms should be represented by classes and not by functions for a couple of reasons that are explained below.
Further all main classes should adhere to the following structure.

This structure is heavily inspired by the interface of `sklearn` and will follow the developer guide outlined [here](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects) for the most part. Below the important points and differences are summarized. It is still recommended to read through the guide!

From the guide:

- The `__init__` of each class should only be there to set parameters. No parameter validation or any functionality should be performed here.
- Defaults for parameters should be provided in the `__init__`
- The names of the class attributes should be identical with the names used in the `__init__`
- Results and outputs are stored with a trailing underscore (e.g. `filtered_stride_list_`
- All algorithms of the same type should have a consistent interface with (as far as possible), identical input parameters to allow drop-in replacements of different algorithms
- Each type of algorithm has one (or multiple) "action" methods with a descriptive name. These *action* methods take the actual data as input and will produce results.
- All *action* methods just return `self` (the object itself)



Deviations from the guide:

- All classes should store the data passed in the "action" step in the class object unless the amount of data would result in an unreasonable performance issue. Ideally this should be a reference and not a copy of the data! This allows to path the final object as a whole to helper functions, that e.g. can visualize in and outputs.
- All methods should take care that they do not modify the original data passed to the function. If required a copy of the data can be created, but **not** stored in the object.
- All classes should have a `_validate(self)` method that handles validation of all the parameters. This function should be called during the "action" (or whenever the parameters are first needed).
Don't overdue the validation and focus on logical validation (e.g. a value can not be larger than x) and not on type validation. For type validation, we should trust that Python provides the correct error message once an invalid step is performed.
- All classes should inherent from a BaseClass specific to their type that implements common functionality and enforces the interface.


#### Example class structure

```python

class EventDetection(BaseEventDetection):
    parameter_one: int
    parameter_two: str
    window_length: float


    data: np.ndarray

    events_: np.ndarray

    def __init__(self, parameter_one: int = 1, parameter_two: Optional[str] = None, window_length: float = 4.):
       self.parameter_one = parameter_one
       self.parameter_two = parameter_two
       self.window_length = window_length


   def detect(self, data: np.ndarray):
       """This is the actual action class."""
       self.data = data
       # Do some calculations
       self.events_ = ...


   def _validate(self):
       """Validate all parameters here."""

       if parameter_two is None:
           raise ValueError("`parameter_two` should a real value")









