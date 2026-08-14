import awkward as ak
import numpy as np
import pytest

from mlpf.data.key4hep.postprocessing import hits_to_features


HIT_FEATURES = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]


def make_hit_data(collection):
    values = {
        "type": [0, 0],
        "cellID": [1, 2],
        "energy": [0.1, 0.2],
        "energyError": [0.0, 0.0],
        "time": [0.0, 0.0],
        "position.x": [1.0, 2.0],
        "position.y": [0.0, 0.0],
        "position.z": [1.0, 2.0],
    }
    return ak.Array({f"{collection}.{feature}": [value] for feature, value in values.items()})


@pytest.mark.parametrize(
    ("collection", "expected_subdetector", "expected_elemtype"),
    [
        ("VXDTrackerHits", 3, 1),
        ("ITrackerEndcapHits", 3, 1),
        ("ECALBarrel", 0, 2),
        ("HCALBarrel", 1, 2),
        ("MUON", 2, 2),
    ],
)
def test_hit_elemtype_follows_subdetector(collection, expected_subdetector, expected_elemtype):
    features = hits_to_features(make_hit_data(collection), 0, collection, HIT_FEATURES)

    np.testing.assert_array_equal(features["subdetector"], [expected_subdetector, expected_subdetector])
    np.testing.assert_array_equal(features["elemtype"], [expected_elemtype, expected_elemtype])
