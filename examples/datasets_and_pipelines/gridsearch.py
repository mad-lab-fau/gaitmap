import pandas as pd
from sklearn.model_selection import ParameterGrid

from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import SimplePipeline, GridSearch
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_right_foot_to_fbf, convert_left_foot_to_fbf
from gaitmap.utils.datatype_helper import SingleSensorStrideList


class MyDataset(Dataset):
    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def data(self):
        self.assert_is_single(None, "data")
        return get_healthy_example_imu_data()[self.index.iloc[0]["foot"] + "_sensor"]

    @property
    def segmented_stride_list_(self):
        self.assert_is_single(None, "data")
        return get_healthy_example_stride_borders()[self.index.iloc[0]["foot"] + "_sensor"].set_index("s_id")

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"participant": ["test", "test"], "foot": ["left", "right"]})


class MyPipeline(SimplePipeline):
    max_cost: float

    segmented_stride_list_: SingleSensorStrideList

    def __init__(self, max_cost: float = 3):
        self.max_cost = max_cost

    def run(self, dataset_single: MyDataset):
        converter = {"left": convert_left_foot_to_fbf, "right": convert_right_foot_to_fbf}
        data = converter[dataset_single.groups[0][1]](dataset_single.data)

        dtw = BarthDtw(max_cost=self.max_cost)
        dtw.segment(data, dataset_single.sampling_rate_hz)

        self.segmented_stride_list_ = dtw.stride_list_
        return self


def score(pipeline: MyPipeline, dataset_single: MyDataset):
    pipeline.run(dataset_single)
    matches_df = evaluate_segmented_stride_list(
        ground_truth=dataset_single.segmented_stride_list_, segmented_stride_list=pipeline.segmented_stride_list_
    )
    return precision_recall_f1_score(matches_df)["f1_score"]

parameters = ParameterGrid({"max_cost" :[3, 5]})

pipe = MyPipeline()

gs = GridSearch(pipe, parameters, scoring=score, rank_scorer="f1_score")
gs = gs.optimize(MyDataset())
print(pd.DataFrame(gs.gs_results_))