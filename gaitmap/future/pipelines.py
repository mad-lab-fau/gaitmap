import numpy as np

from gaitmap.base import _BaseSerializable


class SimplePipeline(_BaseSerializable):
    def run(self, datasets_single):
        raise NotImplementedError()

    def score_single(self, datasets_single):
        raise NotImplementedError()

    def score(self, datasets):
        scores = []
        for d in datasets:
            scores.append(self.score_single(d))
        return np.mean(scores)
