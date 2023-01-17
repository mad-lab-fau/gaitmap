# Gaitmap vs Gaitmap-mad

In the gaitmap repos there are two packages: `gaitmap` and `gaitmap_mad`.
The main package `gaitmap` contains the core functionality of the library and most of the algorithms.
The `gaitmap_mad` package contains additional algorithms that were developed by the MaD-Lab in collaboration with 
industry partners.

Below we explain what this means for you:

## As a user

If you do not need the additional algorithms, you can simply install the `gaitmap` package and ignore everything else.
The `gaitmap` package is licensed under the MIT license and can be used freely.

If you need the additional algorithms, you need to install the `gaitmap_mad` package in addition to the `gaitmap` 
package.
The `gaitmap_mad` package is licensed under the AGPL3, which contains some restrictions on the distribution of the 
software and use in user facing services.
Before using `gaitmap_mad` in your project, make sure you understand the licence terms and please contact us in case of 
doubt.

After installation of `gaitmap_mad`, the additional algorithms are available via the expected `gaitmap` import paths.
This means you never need to import directly from the `gaitmap_mad` package.

```python
# Do this:
from gaitmap.event_detection import RamppEventDetection
# Not this:
from gaitmap_mad.event_detection import RamppEventDetection
```

## As a developer

As a developer, simply clone the repo and run `poetry install` in the top-level directory.
The dependencies are set up in a way, that `gaitmap_mad` is installed as a development dependency of `gaitmap`.
This means all changes that you make to `gaitmap_mad` are immediately available in `gaitmap` and vice versa.

By convention, we don't add any dependencies to `gaitmap_mad`.
If one of the `gaitmap_mad` algorithms requires an additional package, we add it to the `gaitmap` package instead to 
ensure full version compatibility.

If you add a new algorithm to `gaitmap_mad`, we recommend to mirror the structure of the `gaitmap` package.
Then within the respective subpackage `__init__` file of the gaitmap package use our `gaitmap_mad` helpers to make the 
new algorithms conditionally available.

Here an example from the `gaitmap.event_detection.__init__` file mixing algorithms from `gaitmap` and `gaitmap_mad`:

```python
from gaitmap.event_detection._herzer_event_detection import HerzerEventDetection
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {"RamppEventDetection", "FilteredRamppEventDetection"}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.event_detection import FilteredRamppEventDetection, RamppEventDetection

__all__ = ["RamppEventDetection", "HerzerEventDetection", "FilteredRamppEventDetection"]
```

This allows you to still import `HerzerEventDetection` without errors, even if `gaitmap_mad` is not installed.
But if `gaitmap_mad` is installed, you can also import `RamppEventDetection` and `FilteredRamppEventDetection` using 
the same import path. 