import numbers
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd


def _aggregate_scores(
    scores: List[Union[Dict[str, numbers.Number], numbers.Number]]
) -> Tuple[Union[Dict[str, numbers.Number], numbers.Number], Union[str, Dict[str, np.ndarray], np.ndarray]]:
    # TODO: This doesn't work if scoring fails.
    if isinstance(scores[0], dict):
        # Invert the dict and calculate the mean per score:
        df = pd.DataFrame.from_records(scores)
        means = df.mean()
        return means.to_dict(), {k: v.to_numpy() for k, v in df.iteritems()}
    return np.mean(scores), np.asarray(scores)


