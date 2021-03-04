from typing import Callable, Dict, Any

import numpy as np
import pandas as pd

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import SimplePipeline


def _aggregate_scores(scores):
    if isinstance(scores[0], dict):
        # Invert the dict and calculate the mean per score:
        df = pd.DataFrame.from_records(scores)
        means = df.mean()
        return means.to_dict(), {k: v.to_numpy() for k, v in df.iteritems()}
    return np.mean(scores), scores


def _multi_score(pipeline: SimplePipeline, scoring: Callable, data: Dataset):
    scores = []
    for d in data:
        # We clone again here, in case any of the parameters were algorithms themself or the score method of the
        # pipeline does strange things.
        scores.append(scoring(pipeline, dataset_single=d))
    return scores


# TODO: If we have a score single method, we could cache it in the context of GridSearch nested in a cv
#       But we need to be careful, that we correctly invalidate the cache, if the score func changes.
def _score(pipeline: SimplePipeline, scoring: Callable, data: Dataset, parameters: Dict[str, Any]):
    pipeline = pipeline.set_params(**parameters)
    pipeline = pipeline.clone()
    # TODO: Perform aggregation of performance here: Aka mean and maybe weighting?
    #       This would allow to return all individual scores for each "dataset_single" object or even all fitted result
    #       objects for each score.
    scores = _multi_score(pipeline, scoring, data)
    return (*_aggregate_scores(scores), data.groups)


def _passthrough_scoring(pipeline, dataset_single):
    return pipeline.score(dataset_single)
