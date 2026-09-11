#!/usr/bin/env python3
"""Validate IDEA k4RecCalorimeter reconstruction contracts.

This script intentionally uses PODIO's object API rather than branch-name
heuristics. Run it in the same Key4HEP environment as the reconstruction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


CHEREN_DIGI = "DRcaloSiPMreadoutDigiHit"
SCINT_DIGI = "DRcaloSiPMreadoutDigiHit_scint"
CHEREN_SIM = "DRcaloSiPMreadoutSimHit"
SCINT_SIM = "DRcaloSiPMreadout_scint"
CHEREN_LINK = "DRcaloSiPMreadoutDigiHit_cheren_link"
SCINT_LINK = "DRcaloSiPMreadoutDigiHit_scint_link"
CELL_TRUTH_LINK = "CaloHitMCParticleLinks"
CLUSTER_TRUTH_LINK = "ClusterMCParticleLinks"
CLUSTERS = "TopoClusterAll"
CLUSTER_CELLS = "TopoClusterAllCells"
MCPARTICLES = "MCParticles"


def object_id(obj) -> tuple[int, int]:
    oid = obj.getObjectID()
    return int(oid.collectionID), int(oid.index)


def collection_ids(collection) -> set[tuple[int, int]]:
    return {object_id(obj) for obj in collection}


def isclose(left: float, right: float, *, scale: float = 1.0) -> bool:
    return math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-6 * max(1.0, scale))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parameter_strings(frame, key: str) -> list[str]:
    values = frame.get_parameter(key)
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_link_endpoints(
    links,
    from_ids: set[tuple[int, int]],
    to_ids: set[tuple[int, int]],
    *,
    label: str,
) -> None:
    link_ids: set[tuple[int, int]] = set()
    for link in links:
        require(link.isAvailable(), f"{label}: unavailable link object")
        link_id = object_id(link)
        require(link_id not in link_ids, f"{label}: duplicate link ObjectID {link_id}")
        link_ids.add(link_id)
        from_id = object_id(link.getFrom())
        to_id = object_id(link.getTo())
        require(from_id in from_ids, f"{label}: invalid from endpoint {from_id}")
        require(to_id in to_ids, f"{label}: invalid to endpoint {to_id}")
        require(math.isfinite(float(link.getWeight())), f"{label}: non-finite weight")


def validate_digi_sim_links(frame, link_name: str, digi_name: str, sim_name: str) -> dict[str, int]:
    links = frame.get(link_name)
    digis = frame.get(digi_name)
    sim_hits = frame.get(sim_name)
    digi_ids = collection_ids(digis)
    sim_ids = collection_ids(sim_hits)
    validate_link_endpoints(links, digi_ids, sim_ids, label=link_name)
    require(len(links) == len(digis), f"{link_name}: {len(links)} links for {len(digis)} digi hits")
    require(len(links) == len(sim_hits), f"{link_name}: {len(links)} links for {len(sim_hits)} sim hits")

    seen_from: set[tuple[int, int]] = set()
    seen_to: set[tuple[int, int]] = set()
    for link in links:
        from_id = object_id(link.getFrom())
        to_id = object_id(link.getTo())
        require(from_id not in seen_from, f"{link_name}: digi endpoint linked more than once: {from_id}")
        require(to_id not in seen_to, f"{link_name}: sim endpoint linked more than once: {to_id}")
        seen_from.add(from_id)
        seen_to.add(to_id)
        require(
            int(link.getFrom().getCellID()) == int(link.getTo().getCellID()),
            f"{link_name}: cellID mismatch for {from_id} -> {to_id}",
        )
    return {"links": len(links), "digis": len(digis), "sim_hits": len(sim_hits)}


def validate_cell_truth_links(frame, *, require_positive_cells: bool) -> tuple[dict[int, dict[tuple[int, int], float]], dict]:
    links = frame.get(CELL_TRUTH_LINK)
    mcparticles = frame.get(MCPARTICLES)
    digi_ids = collection_ids(frame.get(CHEREN_DIGI)) | collection_ids(frame.get(SCINT_DIGI))
    mc_ids = collection_ids(mcparticles)
    validate_link_endpoints(links, digi_ids, mc_ids, label=CELL_TRUTH_LINK)

    by_hit: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_cell: dict[int, dict[tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
    for link in links:
        weight = float(link.getWeight())
        require(weight > 0.0, f"{CELL_TRUTH_LINK}: non-positive weight {weight}")
        hit = link.getFrom()
        if require_positive_cells:
            require(float(hit.getEnergy()) > 0.0, f"{CELL_TRUTH_LINK}: linked cell has non-positive energy")
        by_hit[object_id(hit)].append(weight)
        by_cell[int(hit.getCellID())][object_id(link.getTo())] += weight

    for hit_id, weights in by_hit.items():
        require(isclose(sum(weights), 1.0), f"{CELL_TRUTH_LINK}: weights for hit {hit_id} sum to {sum(weights)}")
    return by_cell, {"links": len(links), "linked_hits": len(by_hit), "linked_cell_ids": len(by_cell)}


def validate_clusters(
    frame,
    metadata_names: list[str],
    cell_truth_by_id: dict[int, dict[tuple[int, int], float]],
    *,
    expect_dual_readout: bool,
    expect_cellid_cluster_truth: bool,
    reversed_inputs: bool,
) -> tuple[list[dict], dict]:
    clusters = frame.get(CLUSTERS)
    copied_cells = frame.get(CLUSTER_CELLS)
    cluster_links = frame.get(CLUSTER_TRUTH_LINK)
    mc_ids = collection_ids(frame.get(MCPARTICLES))
    cluster_ids = collection_ids(clusters)
    validate_link_endpoints(cluster_links, cluster_ids, mc_ids, label=CLUSTER_TRUTH_LINK)

    copied_by_oid = {object_id(cell): cell for cell in copied_cells}
    copied_cell_ids = [int(cell.getCellID()) for cell in copied_cells]
    require(
        len(copied_cell_ids) == len(set(copied_cell_ids)),
        f"{CLUSTER_CELLS}: duplicate detector cellIDs remain after merging",
    )

    input_by_name = {
        CHEREN_DIGI: defaultdict(lambda: [0.0, 0.0]),
        SCINT_DIGI: defaultdict(lambda: [0.0, 0.0]),
    }
    for name, values in input_by_name.items():
        for hit in frame.get(name):
            entry = values[int(hit.getCellID())]
            entry[0] += float(hit.getEnergy())
            entry[1] = math.hypot(entry[1], float(hit.getEnergyError()))

    for cell in copied_cells:
        cell_id = int(cell.getCellID())
        cheren_energy, cheren_error = input_by_name[CHEREN_DIGI][cell_id]
        scint_energy, scint_error = input_by_name[SCINT_DIGI][cell_id]
        if expect_dual_readout:
            expected_energy = cheren_energy + scint_energy
            expected_error = math.hypot(cheren_error, scint_error)
            require(
                isclose(float(cell.getEnergy()), expected_energy, scale=expected_energy),
                f"{CLUSTER_CELLS}: cell {cell_id} energy {cell.getEnergy()} != {expected_energy}",
            )
            require(
                isclose(float(cell.getEnergyError()), expected_error, scale=expected_error),
                f"{CLUSTER_CELLS}: cell {cell_id} error {cell.getEnergyError()} != {expected_error}",
            )

    actual_cluster_links: dict[tuple[int, int], dict[tuple[int, int], float]] = defaultdict(dict)
    for link in cluster_links:
        cluster_id = object_id(link.getFrom())
        particle_id = object_id(link.getTo())
        weight = float(link.getWeight())
        require(weight > 0.0, f"{CLUSTER_TRUTH_LINK}: non-positive weight {weight}")
        require(particle_id not in actual_cluster_links[cluster_id], f"{CLUSTER_TRUTH_LINK}: duplicate particle link")
        actual_cluster_links[cluster_id][particle_id] = weight

    canonical: list[dict] = []
    linked_clusters = 0
    for cluster in clusters:
        cluster_id = object_id(cluster)
        hits = list(cluster.getHits())
        hit_oids = [object_id(hit) for hit in hits]
        require(len(hit_oids) == len(set(hit_oids)), f"{CLUSTERS}: cluster {cluster_id} repeats a copied hit")
        require(all(hit_id in copied_by_oid for hit_id in hit_oids), f"{CLUSTERS}: cluster {cluster_id} has external hit")
        hit_cell_ids = [int(hit.getCellID()) for hit in hits]
        require(len(hit_cell_ids) == len(set(hit_cell_ids)), f"{CLUSTERS}: cluster {cluster_id} repeats a cellID")
        hit_energy = sum(float(hit.getEnergy()) for hit in hits)
        cluster_energy = float(cluster.getEnergy())
        require(
            isclose(cluster_energy, hit_energy, scale=cluster_energy),
            f"{CLUSTERS}: cluster {cluster_id} energy {cluster_energy} != hit sum {hit_energy}",
        )

        shape = [float(value) for value in cluster.getShapeParameters()]
        require(len(shape) == len(metadata_names), f"{CLUSTERS}: shape length does not match metadata")
        named_shape = dict(zip(metadata_names, shape))
        if expect_dual_readout:
            expected_metadata = ["dR_over_E", "energy_cherenkov", "energy_scintillation"]
            if reversed_inputs:
                expected_metadata = ["dR_over_E", "energy_scintillation", "energy_cherenkov"]
            require(metadata_names == expected_metadata, f"unexpected metadata: {metadata_names}")
            cheren_sum = sum(input_by_name[CHEREN_DIGI][cell_id][0] for cell_id in hit_cell_ids)
            scint_sum = sum(input_by_name[SCINT_DIGI][cell_id][0] for cell_id in hit_cell_ids)
            require(isclose(named_shape["energy_cherenkov"], cheren_sum, scale=cheren_sum), f"{CLUSTERS}: Cherenkov shape closure failed")
            require(isclose(named_shape["energy_scintillation"], scint_sum, scale=scint_sum), f"{CLUSTERS}: scintillation shape closure failed")
            require(isclose(cluster_energy, cheren_sum + scint_sum, scale=cluster_energy), f"{CLUSTERS}: channel closure failed")
        else:
            require(metadata_names == ["dR_over_E"], f"unexpected baseline metadata: {metadata_names}")

        actual_weights = actual_cluster_links.get(cluster_id, {})
        if expect_cellid_cluster_truth:
            expected_weights: dict[tuple[int, int], float] = defaultdict(float)
            if cluster_energy > 0.0:
                for hit in hits:
                    for particle_id, weight in cell_truth_by_id.get(int(hit.getCellID()), {}).items():
                        expected_weights[particle_id] += float(hit.getEnergy()) * weight / cluster_energy
            expected_weights = {particle_id: weight for particle_id, weight in expected_weights.items() if weight > 0.0}
            require(
                set(actual_weights) == set(expected_weights),
                f"{CLUSTER_TRUTH_LINK}: particle set mismatch for {cluster_id}",
            )
            for particle_id, expected_weight in expected_weights.items():
                require(
                    isclose(actual_weights[particle_id], expected_weight, scale=expected_weight),
                    f"{CLUSTER_TRUTH_LINK}: {cluster_id}->{particle_id} weight " f"{actual_weights[particle_id]} != {expected_weight}",
                )
        if actual_weights:
            linked_clusters += 1
            require(
                sum(actual_weights.values()) <= 1.0001,
                f"{CLUSTER_TRUTH_LINK}: weights exceed unity for {cluster_id}",
            )

        canonical.append(
            {
                "cell_ids": sorted(hit_cell_ids),
                "energy": cluster_energy,
                "energy_error": float(cluster.getEnergyError()),
                "components": {key: named_shape[key] for key in metadata_names if key.startswith("energy_")},
            }
        )

    canonical.sort(key=lambda item: (item["cell_ids"], item["energy"]))
    return canonical, {
        "clusters": len(clusters),
        "copied_cells": len(copied_cells),
        "cluster_truth_links": len(cluster_links),
        "linked_clusters": linked_clusters,
    }


def validate_file(args: argparse.Namespace) -> dict:
    from podio import root_io

    reader = root_io.Reader(str(args.input))
    metadata_frame = next(iter(reader.get("metadata")))
    metadata_names = parameter_strings(metadata_frame, f"{CLUSTERS}__shapeParameterNames")
    expected_events = args.expected_events
    summaries = []
    canonical_events = []
    for event_number, frame in enumerate(reader.get("events")):
        if args.max_events is not None and event_number >= args.max_events:
            break
        names = set(frame.getAvailableCollections())
        required = {
            CHEREN_DIGI,
            SCINT_DIGI,
            CHEREN_SIM,
            SCINT_SIM,
            SCINT_LINK,
            CELL_TRUTH_LINK,
            CLUSTER_TRUTH_LINK,
            CLUSTERS,
            CLUSTER_CELLS,
            MCPARTICLES,
        }
        missing = sorted(required - names)
        require(not missing, f"event {event_number}: missing collections {missing}")
        if args.expect_optical_links:
            require(CHEREN_LINK in names, f"event {event_number}: missing {CHEREN_LINK}")
        else:
            require(CHEREN_LINK not in names, f"event {event_number}: unexpected {CHEREN_LINK}")

        event_summary = {"event": event_number}
        event_summary[SCINT_LINK] = validate_digi_sim_links(frame, SCINT_LINK, SCINT_DIGI, SCINT_SIM)
        if args.expect_optical_links:
            event_summary[CHEREN_LINK] = validate_digi_sim_links(frame, CHEREN_LINK, CHEREN_DIGI, CHEREN_SIM)
        cell_truth_by_id, cell_summary = validate_cell_truth_links(frame, require_positive_cells=args.expect_positive_linked_cells)
        canonical, cluster_summary = validate_clusters(
            frame,
            metadata_names,
            cell_truth_by_id,
            expect_dual_readout=args.expect_dual_readout,
            expect_cellid_cluster_truth=args.expect_cellid_cluster_truth,
            reversed_inputs=args.reversed_inputs,
        )
        event_summary["cell_truth"] = cell_summary
        event_summary["clusters"] = cluster_summary
        summaries.append(event_summary)
        canonical_events.append(canonical)

    require(len(summaries) == expected_events, f"event count {len(summaries)} != {expected_events}")
    result = {
        "input": str(args.input),
        "variant": args.variant,
        "events": len(summaries),
        "metadata_names": metadata_names,
        "expect_optical_links": args.expect_optical_links,
        "expect_dual_readout": args.expect_dual_readout,
        "expect_cellid_cluster_truth": args.expect_cellid_cluster_truth,
        "event_summaries": summaries,
        "canonical_clusters": canonical_events,
    }
    args.summary.write_text(json.dumps(result, indent=2))
    print(
        f"PASS {args.variant}: events={len(summaries)} metadata={metadata_names} "
        f"clusters={sum(event['clusters']['clusters'] for event in summaries)} "
        f"cluster_links={sum(event['clusters']['cluster_truth_links'] for event in summaries)}"
    )
    return result


def compare_summaries(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    require(left["events"] == right["events"], "order comparison event-count mismatch")
    require(left["metadata_names"] == ["dR_over_E", "energy_cherenkov", "energy_scintillation"], "left metadata mismatch")
    require(right["metadata_names"] == ["dR_over_E", "energy_scintillation", "energy_cherenkov"], "reversed metadata mismatch")
    for event_number, (left_clusters, right_clusters) in enumerate(zip(left["canonical_clusters"], right["canonical_clusters"])):
        require(len(left_clusters) == len(right_clusters), f"event {event_number}: cluster-count order dependence")
        for left_cluster, right_cluster in zip(left_clusters, right_clusters):
            require(left_cluster["cell_ids"] == right_cluster["cell_ids"], f"event {event_number}: cluster membership order dependence")
            require(
                isclose(left_cluster["energy"], right_cluster["energy"], scale=left_cluster["energy"]),
                f"event {event_number}: cluster-energy order dependence",
            )
            require(
                isclose(left_cluster["energy_error"], right_cluster["energy_error"], scale=left_cluster["energy_error"]),
                f"event {event_number}: cluster-error order dependence",
            )
            require(left_cluster["components"].keys() == right_cluster["components"].keys(), f"event {event_number}: component-name mismatch")
            for name in left_cluster["components"]:
                require(
                    isclose(left_cluster["components"][name], right_cluster["components"][name], scale=left_cluster["components"][name]),
                    f"event {event_number}: {name} order dependence",
                )
    print(f"PASS input-order invariance: {args.left} == {args.right}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one reconstruction output")
    validate.add_argument("input", type=Path)
    validate.add_argument("--variant", required=True)
    validate.add_argument("--expected-events", type=int, required=True)
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--max-events", type=int)
    validate.add_argument("--expect-optical-links", action="store_true")
    validate.add_argument("--expect-dual-readout", action="store_true")
    validate.add_argument("--expect-cellid-cluster-truth", action="store_true")
    validate.add_argument("--expect-positive-linked-cells", action="store_true")
    validate.add_argument("--reversed-inputs", action="store_true")

    compare = subparsers.add_parser("compare-order", help="compare normal and reversed dual-readout inputs")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        validate_file(args)
    else:
        compare_summaries(args)


if __name__ == "__main__":
    main()
