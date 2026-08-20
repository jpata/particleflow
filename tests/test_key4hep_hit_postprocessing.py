import awkward as ak
import numpy as np
import pytest

from mlpf.conf import EDM4HEP
from mlpf.data.key4hep.postprocessing import decode_cellid_field, hits_to_features, parse_cellid_encoding
from mlpf.heptfds.edm4hep_utils.utils_hits import X_FEATURES


HIT_FEATURES = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]
TRACKER_ENCODING = "system:5,side:-2,layer:6,module:11,sensor:8"


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
    features = hits_to_features(make_hit_data(collection), 0, collection, HIT_FEATURES, TRACKER_ENCODING)

    np.testing.assert_array_equal(features["subdetector"], [expected_subdetector, expected_subdetector])
    np.testing.assert_array_equal(features["elemtype"], [expected_elemtype, expected_elemtype])


def test_cellid_encoding_parser_supports_implicit_and_explicit_offsets():
    expected = {"system": (0, 5), "side": (5, -2), "layer": (7, 6), "module": (13, 11), "sensor": (24, 8)}

    assert parse_cellid_encoding(TRACKER_ENCODING) == expected
    assert parse_cellid_encoding("system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8") == expected


def test_tracker_surface_fields_are_decoded_from_cellid():
    system = np.array([1, 4, 6], dtype=np.uint64)
    side = np.array([0, -1, 1], dtype=np.int64)
    layer = np.array([5, 3, 2], dtype=np.uint64)
    encoded_side = np.where(side < 0, side + 4, side).astype(np.uint64)
    cellids = system | (encoded_side << np.uint64(5)) | (layer << np.uint64(7))
    hit_data = make_hit_data("ITrackerEndcapHits")
    hit_data = ak.with_field(hit_data, ak.Array([cellids[:2].tolist()]), "ITrackerEndcapHits.cellID")

    features = hits_to_features(hit_data, 0, "ITrackerEndcapHits", HIT_FEATURES, TRACKER_ENCODING)

    np.testing.assert_array_equal(features["system"], system[:2])
    np.testing.assert_array_equal(features["side"], side[:2])
    np.testing.assert_array_equal(features["layer"], layer[:2])
    np.testing.assert_array_equal(decode_cellid_field(cellids, TRACKER_ENCODING, "side"), side)


def test_tfds_hit_schema_retains_detector_surface_fields():
    assert EDM4HEP.HitFeatures.get_names()[-3:] == ["system", "side", "layer"]
    assert X_FEATURES[-3:] == ["system", "side", "layer"]
