"""Exercise core-only imports in an isolated interpreter."""

import subprocess
import sys


def test_evaluation_imports_without_gaitmap_mad():
    """Core evaluation remains usable when optional algorithm exports are unavailable."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["gaitmap_mad"] = None
from gaitmap.event_detection import evaluation as event_evaluation
from gaitmap.stride_segmentation import evaluation as stride_evaluation
import gaitmap.event_detection as events
import gaitmap.stride_segmentation as strides
from gaitmap.utils.exceptions import GaitmapMadImportError

for module, algorithm in [(events, "RamppEventDetection"), (strides, "BarthDtw")]:
    try:
        getattr(module, "unavailable_attribute")
    except AttributeError:
        pass
    else:
        raise AssertionError("Unknown package attributes must raise AttributeError")
    try:
        getattr(module, algorithm)
    except GaitmapMadImportError:
        pass
    else:
        raise AssertionError("Unavailable optional algorithms must explain how to install them")
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
