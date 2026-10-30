"""Compare paired CLD single-photon ECAL readout pilot ROOT files.

Usage: .venv/bin/python scripts/cld_photon_readout/validate_pair.py \
    --baseline /path/to/photon_ecal_5p1mm_key4hep_2025-05-29 \
    --fine /path/to/photon_ecal_2mm_key4hep_2025-05-29 \
    --output /path/to/comparison.json
"""

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


SAMPLE = "gun_gamma_10gev"
FIELDS = {
    "pdg": ("MCParticles", "PDG"),
    "status": ("MCParticles", "generatorStatus"),
    "px": ("MCParticles", "momentum.x"),
    "py": ("MCParticles", "momentum.y"),
    "pz": ("MCParticles", "momentum.z"),
    "sim_barrel": ("ECalBarrelCollection", "energy"),
    "sim_endcap": ("ECalEndcapCollection", "energy"),
    "digi_barrel": ("ECALBarrel", "energy"),
    "digi_endcap": ("ECALEndcap", "energy"),
    "pfo_pdg": ("PandoraPFOs", "PDG"),
    "pfo_energy": ("PandoraPFOs", "energy"),
}


def files_by_seed(workspace: Path) -> dict[int, Path]:
    root_dir = workspace / "gen" / SAMPLE / "root"
    prefix = f"reco_{SAMPLE}_"
    paths = sorted(root_dir.glob(f"{prefix}*.root"))
    if not paths:
        raise FileNotFoundError(f"No reconstructed photon files in {root_dir}")
    return {int(path.stem.removeprefix(prefix)): path for path in paths}


def read_events(path: Path) -> dict:
    # The default FSSpecSource can stall on this release's PODIO ROOT files.
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as root_file:
        tree = root_file["events"]
        arrays = {key: tree[f"{collection}/{collection}.{field}"].array(library="ak") for key, (collection, field) in FIELDS.items()}

    primary = (arrays["pdg"] == 22) & (arrays["status"] == 1)
    primary_counts = ak.to_numpy(ak.sum(primary, axis=1))
    if not np.all(primary_counts == 1):
        raise ValueError(f"{path}: expected one generator-status-1 photon per event; got {primary_counts.tolist()}")
    momentum = np.column_stack([ak.to_numpy(ak.flatten(arrays[axis][primary])) for axis in ("px", "py", "pz")])

    def counts_and_energy(barrel: str, endcap: str) -> tuple[np.ndarray, np.ndarray]:
        count = ak.to_numpy(ak.num(arrays[barrel]) + ak.num(arrays[endcap]))
        energy = ak.to_numpy(ak.sum(arrays[barrel], axis=1) + ak.sum(arrays[endcap], axis=1))
        return count, energy

    sim_count, sim_energy = counts_and_energy("sim_barrel", "sim_endcap")
    digi_count, digi_energy = counts_and_energy("digi_barrel", "digi_endcap")
    photon_pfo_energy = np.asarray(
        [
            max((float(e) for p, e in zip(pdgs, energies) if p == 22), default=0.0)
            for pdgs, energies in zip(ak.to_list(arrays["pfo_pdg"]), ak.to_list(arrays["pfo_energy"]))
        ],
        dtype=np.float64,
    )
    primary_energy = np.linalg.norm(momentum, axis=1)
    return {
        "momentum": momentum,
        "primary_energy": primary_energy,
        "sim_count": sim_count,
        "sim_energy": sim_energy,
        "digi_count": digi_count,
        "digi_energy": digi_energy,
        "photon_pfo_energy": photon_pfo_energy,
        "photon_pfo_response": np.divide(
            photon_pfo_energy,
            primary_energy,
            out=np.zeros_like(photon_pfo_energy),
            where=primary_energy > 0,
        ),
    }


def summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_files = files_by_seed(args.baseline)
    fine_files = files_by_seed(args.fine)
    if baseline_files.keys() != fine_files.keys():
        raise ValueError("The two workspaces have different seed sets")

    collected = {
        arm: {
            key: [] for key in ("primary_energy", "sim_count", "sim_energy", "digi_count", "digi_energy", "photon_pfo_energy", "photon_pfo_response")
        }
        for arm in ("baseline", "fine")
    }
    event_count = 0
    max_primary_difference = 0.0
    for seed in sorted(baseline_files):
        baseline = read_events(baseline_files[seed])
        fine = read_events(fine_files[seed])
        if baseline["momentum"].shape != fine["momentum"].shape:
            raise ValueError(f"Seed {seed}: different event counts")
        difference = np.abs(baseline["momentum"] - fine["momentum"])
        max_primary_difference = max(max_primary_difference, float(np.max(difference)))
        if not np.allclose(baseline["momentum"], fine["momentum"], rtol=0, atol=1e-5):
            raise ValueError(f"Seed {seed}: photon primary momenta differ between readouts")
        event_count += len(baseline["momentum"])
        for arm, events in (("baseline", baseline), ("fine", fine)):
            for key in collected[arm]:
                collected[arm][key].append(events[key])

    report = {
        "sample": SAMPLE,
        "seeds": sorted(baseline_files),
        "events": event_count,
        "max_primary_momentum_difference_GeV": max_primary_difference,
        "arms": {},
    }
    for arm, fields in collected.items():
        values = {key: np.concatenate(chunks) for key, chunks in fields.items()}
        report["arms"][arm] = {key: summary(value) for key, value in values.items()}
        report["arms"][arm]["photon_pfo_found_fraction"] = float(np.mean(values["photon_pfo_energy"] > 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
