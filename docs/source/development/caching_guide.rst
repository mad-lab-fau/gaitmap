.. _caching_dev_guide:

===================
Caching - Dev Guide
===================

Some of the gaitmap algorithms support caching intermediate outputs with `joblib.Memory`.
This guide explains how to best implement caching support for a new algorithm.

In general caching is implemented by adding an additional parameter called `memory` to a class.
This instance should be passed through to the main calculations so that they can be cached.
If done correctly, smart caching can cache the outputs of multiple steps - for example within the action method -
individually.
This ensures that each cached step is only recalculated if the parameters and inputs that are relevant for this specific
step change.
To not wrap every calculation in a separate caching step, it is advisable to first check, which parts are most
performance critical and focus the caching efforts on them.
Further, it should be considered, which groups of parameters are typically changed together in the actual application.

To avoid pitfalls associated with caching class methods, you must **always cache pure functions** when using smart
caching.
This means you should extract the time consuming calculation steps out of the class and then generate a cached version
of this function within the class body when you need it.


How to make a class cachable?
-----------------------------------
As mentioned above, you should first extract the key parts you want to cache into pure functions.
This is a **must** for this type of caching!
Then you can add an additional keyword argument to the `__init__` called `memory` and has the type `Optional[Memory]`.
This parameter should be stored on the class using the same name.

Then you should restructure the code so that there is a clear entry point for caching.
This can be the `calc_single_sensor` methods a lot of algorithms have.
This method should have a parameter called "memory", which should still expect `Optional[Memory]` as input type.
This method would then be called by the parent method (e.g. the action method) with
`method_name(..., memory=self.memory)`.
At this point you should check, if the parameter is `None` and if yes, replace it with an empty memory object: ::

    if not memory:
        memory = Memory(None)

An empty memory object is equivalent to not caching, but you do not need additional if-checks to see if caching is
active later in your code.
This local variable can then be passed to further methods down the line ideally using the call signature
`memory: Memory` for consistency.

Because, the important calculations are extracted into pure functions, their functionality can not be easily modified
by daughter classes anymore.
One workaround would be to create a static reference to the function in the class definition (aka make it a class
attribute), or create a method that returns a reference to the function.
In both cases, it is possible to overwrite the attribute or method in the subclass and replace it with a different
function.
As an example of this approach see `BaseDtw`.
