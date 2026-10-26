"""Validate the opt-in tracks + calorimeter-hit Parquet target contract."""

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from mlpf.conf import ParticleFeatures


def validate(path):
    fields = ParticleFeatures.get_names()
    pn, energy, jet = (fields.index(key) for key in ("particle_number", "energy", "jet_idx"))
    data = ak.from_parquet(path)
    # Key4hep writes a record of event arrays; ColliderML writes an event array.
    if isinstance(data, ak.Record):
        data = ak.Array({name: data[name] for name in data.fields})
    report = {
        "path": str(path),
        "events": len(data),
        "targets": 0,
        "legacy_targets": 0,
        "track_representatives": 0,
        "calo_representatives": 0,
        "energy_accounting": np.zeros(4),
    }
    for event in data:
        target = np.asarray(ak.to_numpy(event.track_calo_target_particles)).reshape(-1, len(fields))
        track = np.asarray(ak.to_numpy(event.ytarget_track_calo_track)).reshape(-1, len(fields))
        calo = np.asarray(ak.to_numpy(event.ytarget_track_calo_hit)).reshape(-1, len(fields))
        assert len(track) == len(event.X_track)
        input_calo = event.X_track_calo_hit if "X_track_calo_hit" in event.fields else event.X_hit_calo
        assert len(calo) == len(input_calo)
        rows = np.concatenate((track, calo))
        representatives = rows[rows[:, 0] != 0]
        assert np.isfinite(rows).all() and np.isfinite(target).all()
        np.testing.assert_array_equal(np.sort(representatives[:, pn]), np.arange(1, len(target) + 1))
        assert np.all((rows[:, pn] >= 0) & (rows[:, pn] <= len(target)))
        for row in representatives:
            np.testing.assert_array_equal(row, target[int(row[pn]) - 1])
        assert np.all((target[:, jet] >= -1) & (target[:, jet] < len(event.targetjet_track_calo)))
        mapping = ak.to_numpy(event.track_calo_original_to_target)
        assert np.all((mapping >= -1) & (mapping < len(target)))
        accounting = ak.to_numpy(event.track_calo_energy_accounting)
        np.testing.assert_allclose(accounting[0], accounting[1:].sum(), rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(target[:, energy].astype(float).sum(), accounting[3], rtol=1e-6, atol=1e-5)
        report["targets"] += len(target)
        report["legacy_targets"] += int(np.count_nonzero(ak.to_numpy(event.ytarget_track).reshape(-1, len(fields))[:, 0]))
        report["legacy_targets"] += int(np.count_nonzero(ak.to_numpy(event.ytarget_cluster).reshape(-1, len(fields))[:, 0]))
        report["track_representatives"] += int(np.count_nonzero(track[:, 0]))
        report["calo_representatives"] += int(np.count_nonzero(calo[:, 0]))
        report["energy_accounting"] += accounting
    report["energy_accounting"] = dict(zip(("input", "invisible", "dropped", "target"), report["energy_accounting"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps([validate(path) for path in args.inputs], indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")
