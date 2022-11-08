import numpy as np
import pandas as pd
import pytest

from gaitmap_mad.stride_segmentation.hmm import SimpleSegmentationHMM, PreTrainedRothSegmentationModel
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalitySegmentationModel(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = SimpleSegmentationHMM

    @pytest.fixture()
    def after_action_instance(self) -> SimpleSegmentationHMM:
        hmm = PreTrainedRothSegmentationModel()
        data = pd.DataFrame(np.random.rand(hmm.feature_transform.n_features, 100))
        hmm.predict(data)
        return hmm
