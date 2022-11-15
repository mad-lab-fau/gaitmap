gaitmap.data_transform: Scaler and Feature Transforms for IMU data
==================================================================

.. automodule:: gaitmap.data_transform
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitmap.data_transform


Base classes
------------
.. currentmodule:: gaitmap.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: class_with_private.rst

    BaseTransformer
    BaseFilter
    TrainableTransformerMixin

Higher-level Transformers
-------------------------
.. autosummary::
   :toctree: generated/data_transform
   :template: class_with_private.rst

    GroupedTransformer
    ChainedTransformer
    ParallelTransformer

Simple Transformers and Scalers
-------------------------------
.. autosummary::
   :toctree: generated/data_transform
   :template: class_with_private.rst

    IdentityTransformer
    FixedScaler
    StandardScaler
    TrainableStandardScaler
    AbsMaxScaler
    TrainableAbsMaxScaler
    MinMaxScaler
    TrainableMinMaxScaler

Filter
------
.. autosummary::
   :toctree: generated/data_transform
   :template: class_with_private.rst

    ButterworthFilter
