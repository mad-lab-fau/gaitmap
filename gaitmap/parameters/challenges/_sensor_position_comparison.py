"""Reference parameters on the IC-to-IC intervals used by the sensor-position challenges."""

import pandas as pd

from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.utils.stride_list_conversion import convert_stride_list


def calculate_reference_parameters(datapoint, marker):
    """Slice continuous marker trajectories at the converted reference stride borders."""
    events = convert_stride_list(datapoint.mocap_events_, target_stride_type="ic")
    trajectories = datapoint.marker_position_
    positions = {}
    sensor_events = {}
    for foot, strides in events.items():
        sensor = f"{foot}_sensor"
        marker_positions = trajectories[f"{foot[0]}_{marker}"].rename(columns=lambda axis: f"pos_{axis}")
        if strides.empty:
            positions[sensor] = marker_positions.iloc[:0].set_axis(
                pd.MultiIndex.from_arrays([strides.index[:0], pd.Index([], dtype="int64")], names=["s_id", "sample"])
            )
        else:
            positions[sensor] = pd.concat(
                {
                    stride_id: marker_positions.iloc[int(stride["start"]) : int(stride["end"]) + 1].reset_index(
                        drop=True
                    )
                    for stride_id, stride in strides.iterrows()
                },
                names=["s_id", "sample"],
            )
        sensor_events[sensor] = strides
    temporal = TemporalParameterCalculation(expected_stride_type="ic").calculate(
        stride_event_list=sensor_events, sampling_rate_hz=datapoint.mocap_sampling_rate_hz_
    )
    spatial = SpatialParameterCalculation(expected_stride_type="ic").calculate(
        positions=positions,
        orientations=None,
        stride_event_list=sensor_events,
        sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
    )
    return {
        sensor: pd.concat([temporal.parameters_[sensor], spatial.parameters_[sensor]], axis=1)
        for sensor in sensor_events
    }
