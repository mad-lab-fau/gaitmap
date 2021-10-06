import pandas as pd

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import OptimizablePipeline, SimplePipeline


class DummyPipeline(OptimizablePipeline):
    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized

    def self_optimize(self, dataset: Dataset, **kwargs):
        self.optimized = True
        return self


class DummyDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"value": list(range(5))})


def dummy_single_score_func(pipeline, data_point):
    return data_point.groups[0]


def create_dummy_score_func(name):
    return lambda x, y: getattr(x, name)


def create_dummy_multi_score_func(names):
    return lambda x, y: {"score_1": getattr(x, names[0]), "score_2": getattr(x, names[1])}


def dummy_multi_score_func(pipeline, data_point):
    return {"score_1": data_point.groups[0], "score_2": data_point.groups[0] + 1}


def dummy_error_score_func(pipeline, data_point):
    if data_point.groups[0] in [0, 2, 4]:
        raise ValueError("Dummy Error for {}".format(data_point.groups[0]))
    return data_point.groups[0]


def dummy_error_score_func_multi(pipeline, data_point):
    tmp = dummy_error_score_func(pipeline, data_point)
    return {"score_1": tmp, "score_2": tmp}
