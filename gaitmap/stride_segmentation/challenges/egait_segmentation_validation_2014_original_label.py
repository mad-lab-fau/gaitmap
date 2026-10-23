"""A challenge for stride segmentation from continuous IMU data focused on straight strides.

General Information
-------------------

Dataset
    The `Egait Segmentation Validation 2014 <dataset_info_>`_ dataset [1]_ is used for this challenge
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_).
    It contains 4x10m gait tests from 30 participants (10 controls, 10 Parkinson's disease, 10 geriatric patients) and
    15 simulated @home tests from 5 participants per cohort (5 controls, 5 Parkinson's disease, 5 geriatric patients ).
Sensor System
    All participants wore Shimmer2R (102.4 Hz) IMUs laterally on both shoes.
Reference System
    All trials where recorded with a video camera to support labeling of the strides in the raw data.
    The camera data was used to remove turns and stair strides from the annotations.
    Another set of annotations exists, that contains all strides including turns and stairs.
    All annotations where performed following the stride definition by [1]_.
    However, this challenge used the original annotations that **excluded** turns and stairs.
    For an equivalent challenge using the new annotations, see
    :mod:`~gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014`.

Implementation Recommendations
------------------------------

This challenge version uses the original annotations that **excluded** turns and stairs.
In result, the validation favors algorithms that are very specific to straight strides.
Algorithms that naturally have a higher sensitivity and also detect turns and stairs will be penalized.

Overall the data only contains gait and no other movements.
This should make this a relatively easy dataset for gait segmentation.

References
----------
.. [1] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
       Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence
       Dynamic Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419-40.
       https://doi.org/10.3390/s150306419.

.. _dataset_info: https://www.mad.tf.fau.de/research/datasets/#collapse_18
.. _datasets_example:
    https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/egait_segmentation_validation_2014.html
.. _dataset_download: https://www.mad.tf.fau.de/research/datasets/#collapse_18

"""

from gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014 import (
    Challenge as BaseChallenge,
)
from gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014 import (
    ChallengeDataset,
    SensorNames,
)
from gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014 import (
    final_scorer as _final_scorer,
)


def final_scorer(pipeline, datapoint, tolerance_s=0.03, use_original_labels=True):
    """Score segmentation against the original straight-stride annotations by default."""
    return _final_scorer(pipeline, datapoint, tolerance_s=tolerance_s, use_original_labels=use_original_labels)


class Challenge(BaseChallenge):
    """The EgaitSegmentation Validation Challenge using the original labels.

    This challenge is identical to
    :class:`~gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014.Challenge`,
    with the exception that `use_original_labels` is set to True by default.
    This means that the original labels are used for the reference stride list.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.EgaitSegmentationValidation2014` or a path to a directory containing
        the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

    Other Parameters
    ----------------
    match_tolerance_s
        (Class Constant) The tolerance in seconds that is allowed between the calculated stride borders and the
        reference stride borders.
    use_original_labels
        (Class Constant) If True, the original stride labels are used for the reference stride list.
        The original labels only contain straight strides.
        If False, the new labels are used for the reference stride list.
        The new labels contain all strides including turns and stairs.

    Attributes
    ----------
    cv_results_
        The results of the cross-validation.
        This can be passed directly to the pandas DataFrame constructor to get a dataframe with the results.

    See Also
    --------
    gaitmap.utils.challenges.BaseChallenge : For common parameters and attributes of all challenges.
    gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014.Challenge : The same challenge, but with
        the new labels

    Notes
    -----
    Version History:
        - 1.0.0: Original Version, which had a major bug, that the scorer was still using the new labels.
        - 2.0.0 (2023-09-19): Fixed the bug, that the scorer was still using the new labels.

    """

    VERSION = "2.0.0"

    use_original_labels = True


__all__ = ["Challenge", "ChallengeDataset", "SensorNames", "final_scorer"]
