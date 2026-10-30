"""Check source semantics and reproducibility of reconstruction artwork."""

import numpy as np
from scripts.create_reconstruction_event_assets import extract_scene, particle_records
from scripts.visualize_colliderml import EventTree, adapt_event
from scripts.visualize_key4hep import DETECTORS
from tests.test_detector_event_displays import sample_record


def test_reconstructed_candidates_take_precedence_over_truth():
    branches = {"PandoraPFOs": [], "MCParticles": []}
    values = {
        "PDG": [211, 22, 12],
        "charge": [1, 0, 0],
        "momentum.x": [2, 3, 1],
        "momentum.y": [0, 0, 0],
        "momentum.z": [0, 0, 0],
        "mass": [0.14, 0, 0],
    }
    for field, value in values.items():
        branches[f"PandoraPFOs/PandoraPFOs.{field}"] = value
    records, source, label = particle_records(EventTree(branches, 0, 1), 0, DETECTORS["cld"])
    assert source == "PandoraPFOs"
    assert label == "Pandora reconstructed particles"
    assert [p["kind"] for p in records] == [211, 22]
    assert [p["px"] for p in records] == [2, 3]


def test_scene_sampling_is_reproducible_and_truth_is_labeled():
    record = sample_record()
    record["tracker_hits"] = {"detector": np.zeros(30), "x": np.arange(30), "y": np.zeros(30), "z": np.zeros(30)}
    tree = adapt_event(record, 0, 1)
    a = extract_scene(tree, "colliderml", 0, "fixture", max_hits=5)
    b = extract_scene(tree, "colliderml", 0, "fixture", max_hits=5)
    assert a == b
    hit = next(h for h in a["hits"] if h["collection"] == "TrackerRegion0")
    assert hit["count"] == 30 and len(hit["xy"]) == 5
    assert all(x <= 0 for x, y in hit["xy"])
    assert a["particle_label"] == "Truth proxy: primary leaves"


def test_svg_compaction_preserves_marker_geometry_and_other_paths(tmp_path):
    import re
    import xml.etree.ElementTree as ET
    import matplotlib.pyplot as plt
    from scripts.create_reconstruction_event_assets import compact_svg_circles

    path = tmp_path / "markers.svg"
    fig, ax = plt.subplots()
    ax.scatter([1, 2], [3, 4], s=[9, 25], facecolors="none", edgecolors="purple")
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)
    ns = "{http://www.w3.org/2000/svg}"
    before = ET.parse(path)
    markers = [p for p in before.findall(f".//{ns}path") if "stroke: #800080" in p.get("style", "")]
    bounds = []
    for p in markers:
        coords = np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", p.get("d"))]).reshape(-1, 2)
        bounds.append((coords.min(axis=0), coords.max(axis=0)))
    assert len(bounds) == 2
    compact_svg_circles(path)
    after = ET.parse(path)
    circles = [p for p in after.findall(f".//{ns}circle") if "stroke: #800080" in p.get("style", "")]
    assert len(circles) == 2
    for circle, (lo, hi) in zip(circles, bounds):
        center = np.array([float(circle.get("cx")), float(circle.get("cy"))])
        radius = float(circle.get("r"))
        np.testing.assert_allclose(center - radius, lo, atol=1e-6, rtol=0)
        np.testing.assert_allclose(center + radius, hi, atol=1e-6, rtol=0)
    assert len(after.findall(f".//{ns}path")) == len(before.findall(f".//{ns}path")) - 2
