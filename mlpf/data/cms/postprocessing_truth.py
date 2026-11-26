"""Small, row-aligned PF targets from truth-graph associations.

Calorimeter edges must contain reconstructed-energy *product* overlaps, not the
simulation-energy ``sharedEnergy`` currently stored in HGCal NanoAOD. The CMSSW
exporter ``scripts/cmssw/extract_association_metrics.py --skip-legacy`` supplies
these weights. Tracks use (1 - forward score) * reconstructed momentum.

The assignment deliberately follows postprocessing2.py's simple ordered greedy
policy. Original truth and sparse split/merge provenance are retained separately.
This is a versioned Phase-2 prototype, not a replacement for the CMS TFDS loader.
"""

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLLECTIONS = ("GSF", "Track", "ECAL", "HCAL", "HGCal")
TYPE_CODES = {"Track": 1, "GSF": 6, "ECAL": 4, "HCAL": 5, "HGCal": 12}
X_DTYPE = [(name, "f4") for name in ("typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz")]
Y_NAMES = (
    "pid",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "ispu",
    "generatorStatus",
    "simulatorStatus",
    "cp_to_track",
    "cp_to_cluster",
    "jet_idx",
    "particle_number",
)
Y_DTYPE = [(name, "f4") for name in Y_NAMES]
PROVENANCE_DTYPE = [("row", "i4"), ("truth", "i4"), ("fraction", "f8")]


def four_vector(pt, eta, phi, energy):
    return np.array([pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), energy], dtype=np.float64)


def kinematics(p4):
    px, py, pz, energy = p4
    pt = math.hypot(px, py)
    return pt, math.asinh(pz / pt) if pt else 0.0, math.atan2(py, px), float(energy)


def delta_r(p4a, p4b):
    _, eta_a, phi_a, _ = kinematics(p4a)
    _, eta_b, phi_b, _ = kinematics(p4b)
    return math.hypot(eta_a - eta_b, math.remainder(phi_a - phi_b, 2 * math.pi))


@dataclass(frozen=True)
class Particle:
    truth_id: int
    pdg_id: int
    charge: float
    p4: np.ndarray
    ispu: float = 0.0


@dataclass(frozen=True)
class Element:
    collection: str
    index: int
    p4: np.ndarray
    charge: float = 0.0


@dataclass(frozen=True)
class Link:
    truth: int
    element: int
    weight: float
    purity: float = 1.0


@dataclass
class Event:
    event_id: tuple
    truth: list
    elements: list
    links: list
    gen: list
    gen_ancestors: list


def accepted_links(event, track_min_purity=0.75):
    """Keep measured overlaps; GSF anchors are reserved for electrons."""
    result = {}
    for link in event.links:
        if not (0 <= link.truth < len(event.truth) and 0 <= link.element < len(event.elements)):
            raise ValueError("Association index outside event tables")
        if not math.isfinite(link.weight) or link.weight < 0 or not math.isfinite(link.purity) or not 0 <= link.purity <= 1:
            raise ValueError("Association weights/purities must be finite and weights nonnegative")
        key = (link.truth, link.element)
        if key in result:
            raise ValueError("Duplicate truth/element association")
        element = event.elements[link.element]
        particle = event.truth[link.truth]
        if link.weight == 0:
            continue
        if element.collection == "GSF" and abs(particle.pdg_id) != 11:
            continue
        if element.collection in ("Track", "GSF") and link.purity <= track_min_purity:
            continue
        result[key] = link.weight
    return result


def _fragments(event, weights, split_dr):
    """A fragment is (original truth row, four-vector fraction, allowed edges)."""
    fragments, split_truth = [], []
    for truth, particle in enumerate(event.truth):
        edges = {element: weight for (root, element), weight in weights.items() if root == truth}
        tracks = sorted(
            element
            for element in edges
            if event.elements[element].collection == "Track" and delta_r(particle.p4, event.elements[element].p4) < split_dr
        )
        if abs(particle.pdg_id) != 11 and len(tracks) > 1:
            total = sum(edges[element] for element in tracks)
            fragments.extend((truth, edges[element] / total, {element: edges[element]}) for element in tracks)
            split_truth.append(truth)
        elif edges:
            fragments.append((truth, 1.0, edges))
    return fragments, split_truth


def _assign(event, fragments):
    """One distinct anchor first; leftovers merge onto their strongest edge."""
    assigned = {}
    rows = [[] for _ in event.elements]
    for collection in COLLECTIONS:
        elements = sorted(
            (i for i, element in enumerate(event.elements) if element.collection == collection),
            key=lambda i: (-kinematics(event.elements[i].p4)[0], i),
        )
        for element in elements:
            choices = [(edges[element], i) for i, (_, _, edges) in enumerate(fragments) if i not in assigned and element in edges]
            if choices:
                _, fragment = max(choices, key=lambda item: (item[0], -item[1]))
                assigned[fragment] = element
                rows[element].append(fragment)
    leftovers = sorted(
        (i for i in range(len(fragments)) if i not in assigned),
        key=lambda i: (-kinematics(event.truth[fragments[i][0]].p4)[0] * fragments[i][1], i),
    )
    for fragment in leftovers:
        edges = fragments[fragment][2]
        element = max(edges, key=lambda i: (edges[i], -i))
        rows[element].append(fragment)
    return rows


def process_event(event, track_min_purity=0.75, split_dr=0.4):
    """Return simple PF tensors and explicit original-truth contributions."""
    if not 0 <= track_min_purity < 1 or not math.isfinite(split_dr) or split_dr <= 0:
        raise ValueError("Invalid track purity or splitting radius")
    if len({particle.truth_id for particle in event.truth}) != len(event.truth):
        raise ValueError("Truth particle IDs must be unique within an event")
    if len({(element.collection, element.index) for element in event.elements}) != len(event.elements):
        raise ValueError("Input collection/index identities must be unique within an event")
    for particle in event.truth:
        if not np.all(np.isfinite(particle.p4)) or particle.p4[3] <= 0:
            raise ValueError("Truth four-vectors must be finite with positive energy")
    for element in event.elements:
        if element.collection not in COLLECTIONS or not np.all(np.isfinite(element.p4)):
            raise ValueError("Invalid element collection or four-vector")
    weights = accepted_links(event, track_min_purity)
    fragments, split_truth = _fragments(event, weights, split_dr)
    rows = _assign(event, fragments)
    x = np.zeros(len(rows), dtype=X_DTYPE)
    y = np.zeros(len(rows), dtype=Y_DTYPE)
    p4 = np.zeros((len(rows), 4), dtype=np.float64)
    target_id = np.zeros(len(rows), dtype=np.int64)
    mixed = np.zeros(len(rows), dtype=bool)
    contributions = []
    support = np.zeros((len(event.truth), 2))
    for (truth, element), weight in weights.items():
        support[truth, int(event.elements[element].collection not in ("Track", "GSF"))] += weight
    for row, element in enumerate(event.elements):
        pt, eta, phi, energy = kinematics(element.p4)
        x[row] = (TYPE_CODES[element.collection], pt, eta, phi, energy, element.charge, *element.p4[:3])
        parents = {}
        for fragment in rows[row]:
            truth, fraction, _ = fragments[fragment]
            parents[truth] = parents.get(truth, 0.0) + fraction
        if not parents:
            continue
        for truth, fraction in sorted(parents.items()):
            p4[row] += fraction * event.truth[truth].p4
            contributions.append((row, truth, fraction))
        dominant = max(parents, key=lambda truth: (parents[truth] * event.truth[truth].p4[3], -truth))
        particle = event.truth[dominant]
        pt, eta, phi, energy = kinematics(p4[row])
        ispu = sum(fraction * event.truth[truth].p4[3] * event.truth[truth].ispu for truth, fraction in parents.items()) / energy
        track_support, calo_support = sum((fraction * support[truth] for truth, fraction in parents.items()), start=np.zeros(2))
        target_id[row] = row + 1
        mixed[row] = len({(event.truth[t].pdg_id, event.truth[t].charge) for t in parents}) > 1
        y[row] = (
            abs(particle.pdg_id),
            particle.charge,
            pt,
            eta,
            math.sin(phi),
            math.cos(phi),
            energy,
            ispu,
            0,
            1 if any(t in split_truth for t in parents) else 2,
            track_support,
            calo_support,
            -1,
            target_id[row],
        )
    provenance = np.array(contributions, dtype=PROVENANCE_DTYPE)
    fractions = np.bincount(provenance["truth"], weights=provenance["fraction"], minlength=len(event.truth))
    represented = fractions > 0
    if not np.allclose(fractions[represented], 1, atol=1e-12, rtol=0):
        raise AssertionError("Split/merge provenance does not preserve truth fractions")
    original_sum = sum((p.p4 for p, keep in zip(event.truth, represented) if keep), start=np.zeros(4))
    if not np.allclose(p4.sum(axis=0), original_sum, rtol=1e-12, atol=1e-10):
        raise AssertionError("Split/merge assignment does not preserve four-momentum")
    return {
        "schema": "cms_truth_graph_v1",
        "event_id": event.event_id,
        "Xelem": x,
        "ytarget": y,
        "target_p4": p4,
        "target_id": target_id,
        "mixed": mixed,
        "provenance": provenance,
        "truth": event.truth,
        "elements": event.elements,
        "gen": event.gen,
        "gen_ancestors": event.gen_ancestors,
        "split_truth": split_truth,
        "unmatched_truth": np.flatnonzero(~represented).tolist(),
        "merged_rows": [i for i, fragments_here in enumerate(rows) if len({fragments[f][0] for f in fragments_here}) > 1],
    }


def _nano_particles(arrays, entry, prefix, ids):
    result = []
    cartesian = all(f"{prefix}_{key}" in arrays for key in ("px", "py", "pz"))
    coordinates = ("px", "py", "pz", "energy") if cartesian else ("pt", "eta", "phi", "energy")
    values = {key: np.asarray(arrays[f"{prefix}_{key}"][entry]) for key in ("pdgId", *coordinates)}
    if prefix == "TruthBranch":
        values.update({key: np.asarray(arrays[f"{prefix}_{key}"][entry]) for key in ("charge", "isSignal")})
    for i, truth_id in enumerate(ids):
        pdg_id = int(values["pdgId"][i])
        charge = float(values["charge"][i]) if prefix == "TruthBranch" else 0.0
        # Cartesian columns avoid amplification of float32 phi rounding near
        # px=0 or py=0; keep the angular fallback for older NanoAOD schemas.
        p4 = (
            np.array([values[key][i] for key in coordinates], dtype=np.float64)
            if cartesian
            else four_vector(*(float(values[key][i]) for key in coordinates))
        )
        ispu = float(not values["isSignal"][i]) if prefix == "TruthBranch" else 0.0
        result.append(Particle(int(truth_id), pdg_id, charge, p4, ispu))
    return result


def _nano_elements(arrays, entry, tracksters):
    result, offsets = [], {}
    for collection, prefix in (
        ("Track", "GeneralTrack"),
        ("GSF", "GSFTrack"),
        ("ECAL", "PFClusterECAL"),
        ("HCAL", "PFClusterHCAL"),
        ("HGCal", tracksters),
    ):
        offsets[collection] = len(result)
        for index in range(int(arrays[f"n{prefix}"][entry])):
            if collection in ("Track", "GSF"):
                pt, eta, phi, energy = (float(arrays[f"{prefix}_{key}"][entry][index]) for key in ("pt", "eta", "phi", "p"))
                charge = float(arrays[f"{prefix}_charge"][entry][index])
            else:
                energy_key, eta_key, phi_key = (
                    ("raw_energy", "barycenter_eta", "barycenter_phi") if collection == "HGCal" else ("energy", "eta", "phi")
                )
                energy, eta, phi = (float(arrays[f"{prefix}_{key}"][entry][index]) for key in (energy_key, eta_key, phi_key))
                pt, charge = energy / math.cosh(eta), 0.0
            result.append(Element(collection, index, four_vector(pt, eta, phi, energy), charge))
    return result, offsets


def _nano_track_links(arrays, entry, offsets):
    links = []
    for collection, prefix in (("Track", "Track2TruthBranch"), ("GSF", "GSFTrack2TruthBranch")):
        offset = 0
        counts = arrays[f"{prefix}_n{prefix}Links"][entry]
        for element, count in enumerate(counts):
            momentum = float(arrays[f"{'GeneralTrack' if collection == 'Track' else 'GSFTrack'}_p"][entry][element])
            for index in range(offset, offset + int(count)):
                truth = int(arrays[f"{prefix}Links_index"][entry][index])
                purity = 1 - float(arrays[f"{prefix}Links_score"][entry][index])
                links.append(Link(truth, offsets[collection] + element, purity * momentum, purity))
            offset += int(count)
    return links


def read_events(nano_path, metric_path, tracksters="ticlTracksterLinks"):
    """Join NanoAOD features to calibrated product weights with identity checks."""
    import uproot

    payload = json.loads(Path(metric_path).read_text())
    metadata = {int(event["event"]): event for event in payload["events"]}
    if len(metadata) != len(payload["events"]):
        raise ValueError("Metric sidecar event numbers must be unique")
    rows = {}
    for row in payload["rows"]:
        rows.setdefault(int(row["event"]), []).append(row)
    with uproot.open(nano_path) as root:
        arrays = root["Events"].arrays(library="ak", how=dict)
        for entry in range(root["Events"].num_entries):
            event_id = tuple(int(arrays[key][entry]) for key in ("run", "luminosityBlock", "event"))
            meta = metadata[event_id[2]]
            if (int(meta.get("run", event_id[0])), int(meta.get("lumi", event_id[1]))) != event_id[:2]:
                raise ValueError("NanoAOD and metric event identity differ")
            if len(meta["roots"]) != int(arrays["nTruthBranch"][entry]):
                raise ValueError("NanoAOD and metric truth-root counts differ")
            truth = _nano_particles(arrays, entry, "TruthBranch", meta["roots"])
            if "root_p4" in meta:
                if len(meta["root_p4"]) != len(truth) or len(meta["root_pdg_ids"]) != len(truth):
                    raise ValueError("Metric truth metadata is incomplete")
                if any(
                    p.pdg_id != pid or not np.allclose(p.p4, p4, rtol=1e-5, atol=1e-6)
                    for p, pid, p4 in zip(truth, meta["root_pdg_ids"], meta["root_p4"])
                ):
                    raise ValueError("NanoAOD and metric truth identities differ")
            elements, offsets = _nano_elements(arrays, entry, tracksters)
            links = _nano_track_links(arrays, entry, offsets)
            for row in rows.get(event_id[2], []):
                collection = "HGCal" if row["collection"] == tracksters else row["collection"]
                if collection not in ("ECAL", "HCAL", "HGCal"):
                    continue
                index = offsets[collection] + int(row["object"])
                if int(row["object"]) < 0 or index >= len(elements) or elements[index].collection != collection:
                    raise ValueError("Metric element index differs from NanoAOD")
                target = int(row["root"])
                if not 0 <= target < len(truth):
                    raise ValueError("Metric truth index is outside NanoAOD")
                if not np.isclose(truth[target].p4[3], row["truth_energy"], rtol=1e-5):
                    raise ValueError("Metric truth row differs from NanoAOD")
                if row["old_graph"] > 0:
                    links.append(Link(target, index, float(row["old_graph"])))
            gen = meta.get("gen_status1")
            if gen is None:
                raise ValueError("Re-export metrics with gen_status1 metadata; the NanoAOD gen table has a pT cut")
            gen = [Particle(int(p["index"]), int(p["pdg_id"]), float(p["charge"]), np.asarray(p["p4"])) for p in gen]
            yield Event(event_id, truth, elements, links, gen, meta.get("gen_ancestors", [[] for _ in truth]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nano", type=Path)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracksters", choices=("ticlTracksterLinks", "ticlTrackstersCLUE3DHigh"), default="ticlTracksterLinks")
    args = parser.parse_args()
    result = [process_event(event) for event in read_events(args.nano, args.associations, args.tracksters)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(result)} events to {args.output}")


if __name__ == "__main__":
    # Keep dataclass pickle names importable when invoked with python -m.
    from mlpf.data.cms.postprocessing_truth import main as module_main

    module_main()
