"""Controlled comparisons to the legacy assignment and status-1 generator truth."""

import contextlib
import io
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from mlpf.data.cms.postprocessing_truth import accepted_links, delta_r, kinematics


def legacy_reference(event):
    """Run the actual old normalizer on identical inputs, targets and weights.

    This isolates assignment from the old ntuplizer's different truth coverage.
    HGCal uses the last legacy calorimeter dispatch slot (11) in this adapter
    only. Both implementations therefore visit it after HCAL. Missing old input
    features are zero; they are not used by the legacy assignment.
    """
    import networkx as nx

    from mlpf.data.cms import postprocessing2 as legacy

    graph = nx.DiGraph()
    weights = accepted_links(event)
    support = np.zeros((len(event.truth), 2))
    for (truth, element), weight in weights.items():
        support[truth, int(event.elements[element].collection not in ("Track", "GSF"))] += weight
    for row, element in enumerate(event.elements):
        pt, eta, phi, energy = kinematics(element.p4)
        features = {name: 0.0 for name in legacy.elem_branches}
        typ = {"Track": 1, "GSF": 6, "ECAL": 4, "HCAL": 5, "HGCal": 11}[element.collection]
        features.update(typ=typ, pt=pt, eta=eta, phi=phi, energy=energy, charge=element.charge)
        features.update(zip(("px", "py", "pz"), element.p4[:3]))
        graph.add_node(("elem", row), **features)
    for row, particle in enumerate(event.truth):
        pt, eta, phi, energy = kinematics(particle.p4)
        graph.add_node(
            ("cp", row),
            pid=abs(particle.pdg_id),
            charge=particle.charge,
            pt=pt,
            eta=eta,
            phi=phi,
            energy=energy,
            ispu=particle.ispu,
            generatorStatus=0,
            simulatorStatus=0,
            cp_to_track=support[row, 0],
            cp_to_cluster=support[row, 1],
            jet_idx=-1,
            particle_number=float(row + 1),
        )
    for (truth, element), weight in weights.items():
        graph.add_edge(("cp", truth), ("elem", element), weight=weight)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        _, _, targets = legacy.prepare_normalized_table(graph, event.event_id[-1])
    return targets, log.getvalue()


def target_vectors(targets):
    return np.column_stack(
        (
            targets["pt"] * targets["cos_phi"],
            targets["pt"] * targets["sin_phi"],
            targets["pt"] * np.sinh(targets["eta"]),
            targets["energy"],
        )
    )


def match_generator(result, max_dr=0.1):
    """Match derived targets to uncut visible status-1 particles by PID/charge/dR."""
    selected = np.flatnonzero(result["target_id"])
    gen = [particle for particle in result["gen"] if abs(particle.pdg_id) not in (12, 14, 16)]
    cost = np.full((len(selected), len(gen)), 1e6)
    for i, row in enumerate(selected):
        parents = result["provenance"]["truth"][result["provenance"]["row"] == row]
        ancestors = {ancestor for parent in parents for ancestor in result["gen_ancestors"][int(parent)]}
        for j, particle in enumerate(gen):
            if particle.truth_id not in ancestors:
                continue
            if abs(particle.pdg_id) != result["ytarget"]["pid"][row] or particle.charge != result["ytarget"]["charge"][row]:
                continue
            distance = delta_r(result["target_p4"][row], particle.p4)
            if distance < max_dr:
                cost[i, j] = distance
    matched = []
    if cost.size:
        targets, generators = linear_sum_assignment(cost)
        for i, j in zip(targets, generators):
            if cost[i, j] == 1e6:
                continue
            row, particle = int(selected[i]), gen[j]
            pt, _, _, energy = kinematics(result["target_p4"][row])
            gen_pt, _, _, gen_energy = kinematics(particle.p4)
            matched.append(
                dict(
                    row=row,
                    gen_id=particle.truth_id,
                    pdg_id=particle.pdg_id,
                    dr=float(cost[i, j]),
                    pt_ratio=pt / gen_pt if gen_pt else None,
                    energy_ratio=energy / gen_energy,
                )
            )
    return matched, gen


def generator_closure(result):
    """Aggregate split/converted targets by their stored status-1 GEN ancestor."""
    by_gen = {particle.truth_id: np.zeros(4) for particle in result["gen"]}
    roots_by_gen = {particle.truth_id: np.zeros(4) for particle in result["gen"]}
    ambiguous_roots = set()
    for truth, particle in enumerate(result["truth"]):
        ancestors = result["gen_ancestors"][truth]
        if len(ancestors) == 1 and ancestors[0] in roots_by_gen:
            roots_by_gen[ancestors[0]] += particle.p4
    for contribution in result["provenance"]:
        truth = int(contribution["truth"])
        ancestors = result["gen_ancestors"][truth]
        if len(ancestors) != 1 or ancestors[0] not in by_gen:
            ambiguous_roots.add(truth)
            continue
        by_gen[ancestors[0]] += contribution["fraction"] * result["truth"][truth].p4
    rows = []
    for particle in result["gen"]:
        if abs(particle.pdg_id) in (12, 14, 16):
            continue
        p4 = by_gen[particle.truth_id]
        rows.append(
            dict(
                gen_id=particle.truth_id,
                pdg_id=particle.pdg_id,
                gen_pt=kinematics(particle.p4)[0],
                gen_eta=kinematics(particle.p4)[1],
                gen_energy=float(particle.p4[3]),
                energy_ratio=float(p4[3] / particle.p4[3]),
                original_root_energy_ratio=float(roots_by_gen[particle.truth_id][3] / particle.p4[3]),
                represented=bool(p4[3] > 0),
                dr=delta_r(p4, particle.p4) if p4[3] > 0 else None,
            )
        )
    return rows, sorted(ambiguous_roots)


def assigned_elements(result):
    rows = []
    for row in np.flatnonzero(result["target_id"]):
        element = result["elements"][row]
        target_pt, _, _, target_energy = kinematics(result["target_p4"][row])
        input_pt, _, _, input_energy = kinematics(element.p4)
        rows.append(
            dict(
                row=int(row),
                collection=element.collection,
                input_index=element.index,
                pdg_id=int(result["ytarget"]["pid"][row]),
                target_charge=float(result["ytarget"]["charge"][row]),
                input_charge=float(element.charge),
                target_pt=target_pt,
                input_pt=input_pt,
                target_energy=target_energy,
                input_energy=input_energy,
                input_pt_over_target=input_pt / target_pt if target_pt else None,
                input_energy_over_target=input_energy / target_energy,
                dr=delta_r(element.p4, result["target_p4"][row]),
            )
        )
    return rows


def compare_event(event, result):
    reference, log = legacy_reference(event)
    old_p4 = target_vectors(reference)
    difference = np.abs(old_p4 - result["target_p4"])
    matched, gen = match_generator(result)
    closure, ambiguous = generator_closure(result)
    labels_equal = bool(np.array_equal(reference["pid"], result["ytarget"]["pid"]))
    charges_equal = bool(np.array_equal(reference["charge"], result["ytarget"]["charge"]))
    unmatched_details = []
    for truth in result["unmatched_truth"]:
        links = [link for link in event.links if link.truth == truth]
        pt, eta, phi, energy = kinematics(event.truth[truth].p4)
        unmatched_details.append(
            dict(
                truth=truth,
                pdg_id=event.truth[truth].pdg_id,
                pt=pt,
                eta=eta,
                phi=phi,
                energy=energy,
                raw_links=len(links),
                gen_ancestors=event.gen_ancestors[truth],
                best_track_purity=max((link.purity for link in links if event.elements[link.element].collection in ("Track", "GSF")), default=None),
            )
        )
    metrics = dict(
        event_id=list(event.event_id),
        inputs=len(event.elements),
        original_truth=len(event.truth),
        targets=int(np.count_nonzero(result["target_id"])),
        legacy_targets=int(np.count_nonzero(reference["pid"])),
        split_truth=len(result["split_truth"]),
        merged_rows=len(result["merged_rows"]),
        unmatched_truth=result["unmatched_truth"],
        same_legacy_p4=bool(np.allclose(old_p4, result["target_p4"], rtol=2e-5, atol=2e-5)),
        max_legacy_p4_difference=float(difference.max()) if difference.size else 0.0,
        same_legacy_pid=labels_equal,
        same_legacy_charge=charges_equal,
        legacy_mismatch_rows=[
            dict(
                row=int(row),
                collection=event.elements[row].collection,
                old_pid=float(reference["pid"][row]),
                new_pid=float(result["ytarget"]["pid"][row]),
                old_charge=float(reference["charge"][row]),
                new_charge=float(result["ytarget"]["charge"][row]),
                max_p4_difference=float(difference[row].max()),
            )
            for row in range(len(event.elements))
            if not np.allclose(old_p4[row], result["target_p4"][row], rtol=2e-5, atol=2e-5)
            or reference["pid"][row] != result["ytarget"]["pid"][row]
            or reference["charge"][row] != result["ytarget"]["charge"][row]
        ],
        mixed_label_merges=int(np.count_nonzero(result["mixed"])),
        gen_status1=len(event.gen),
        visible_gen_status1=len(gen),
        direct_gen_matches=matched,
        generator_closure=closure,
        ambiguous_gen_ancestry=ambiguous,
        assigned_elements=assigned_elements(result),
        unmatched_details=unmatched_details,
    )
    # The prototype deliberately fixes the old mixed-merge PID/charge convention.
    # Classify it only when both dominant parents can be verified from provenance
    # and accepted edge weights; preserve all raw legacy comparisons above.
    charge_convention = []
    weights = accepted_links(event) if metrics["legacy_mismatch_rows"] else {}
    for mismatch in metrics["legacy_mismatch_rows"]:
        row = mismatch["row"]
        parents = result["provenance"][result["provenance"]["row"] == row]
        if len(parents) < 2 or mismatch["old_pid"] != mismatch["new_pid"]:
            continue
        if not np.allclose(old_p4[row], result["target_p4"][row], rtol=2e-5, atol=2e-5):
            continue
        dominant = max(parents, key=lambda c: (float(c["fraction"]) * event.truth[int(c["truth"])].p4[3], -int(c["truth"])))
        energy_parent = int(dominant["truth"])
        parent_weights = {int(c["truth"]): weights.get((int(c["truth"]), row), 0) for c in parents}
        maximum = max(parent_weights.values())
        edge_parents = [truth for truth, weight in parent_weights.items() if weight == maximum]
        particle = event.truth[energy_parent]
        if abs(particle.pdg_id) != mismatch["new_pid"] or particle.charge != mismatch["new_charge"]:
            continue
        if not any(event.truth[truth].charge == mismatch["old_charge"] for truth in edge_parents):
            continue
        charge_convention.append(dict(row=row, energy_parent=energy_parent, strongest_edge_parents=edge_parents))
    explained = {entry["row"] for entry in charge_convention}
    metrics["legacy_charge_convention_rows"] = charge_convention
    metrics["unexpected_legacy_mismatch_rows"] = [row for row in metrics["legacy_mismatch_rows"] if row["row"] not in explained]
    for field in ("pt", "energy"):
        if not math.isfinite(float(result["ytarget"][field].sum())):
            raise AssertionError("Non-finite target output")
    return metrics, reference, log
