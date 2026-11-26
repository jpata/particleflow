"""Direct fiducial GEN status-1 versus assigned-target jet and MET validation."""

from dataclasses import asdict, dataclass

import fastjet
import numpy as np

from mlpf.jet_utils import _match_jets_event


@dataclass(frozen=True)
class ResponseConfig:
    particle_pt_min: float = 0.5
    particle_eta_max: float = 3.2
    jet_radius: float = 0.4
    genjet_pt_min: float = 20.0
    targetjet_pt_min: float = 5.0
    jet_eta_max: float = 3.2
    match_dr: float = 0.2
    met_response_min: float = 20.0

    def __post_init__(self):
        if min(asdict(self).values()) <= 0:
            raise ValueError("Response cuts must be positive")
        if self.jet_eta_max > self.particle_eta_max + 1e-9:
            raise ValueError("Jet-axis acceptance must not exceed constituent acceptance")
        if self.targetjet_pt_min > self.genjet_pt_min:
            raise ValueError("Target matching threshold must not exceed GEN threshold")

    def metadata(self):
        return dict(
            **asdict(self),
            reference="Visible GEN status 1 versus new assigned targets only",
            particles="Same independent particle cuts on both collections; neutrinos excluded; electrons and muons included in jets",
            jets="anti-kT, E-scheme; matched response conditional on unique angular matching, with no response cut; fiducial constituents and axes, edge cones are truncated",
            matching="Maximum-cardinality minimum-total-dR one-to-one matching",
            met="Negative vector sum of fiducial visible particle momenta; fiducial imbalance, not full-acceptance neutrino MET",
            resolution="sigma68=(Q84-Q16)/2; relative response resolution=sigma68/abs(median)",
            uncertainty="68% bootstrap intervals; events resampled as clusters for jet quantities",
        )


def kinematics(vectors):
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 4)
    pt = np.hypot(vectors[:, 0], vectors[:, 1])
    eta = np.arcsinh(vectors[:, 2] / np.maximum(pt, 1e-30))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    return pt, eta, phi


def fiducial(vectors, config):
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 4)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Non-finite particle four-vector")
    pt, eta, _ = kinematics(vectors)
    return vectors[(pt > config.particle_pt_min) & (np.abs(eta) <= config.particle_eta_max + 1e-12)]


def cluster_jets(vectors, min_pt, config):
    if not len(vectors):
        return np.empty((0, 4))
    inputs = [fastjet.PseudoJet(*map(float, p)) for p in vectors]
    sequence = fastjet.ClusterSequence(inputs, fastjet.JetDefinition(fastjet.antikt_algorithm, config.jet_radius))
    jets = fastjet.sorted_by_pt(sequence.inclusive_jets(min_pt))
    p4 = np.array([[j.px(), j.py(), j.pz(), j.E()] for j in jets]).reshape(-1, 4)
    return p4[np.abs(kinematics(p4)[1]) <= config.jet_eta_max + 1e-12]


def met_quantities(gen, target):
    reference = -np.asarray(gen).reshape(-1, 4)[:, :2].sum(axis=0)
    candidate = -np.asarray(target).reshape(-1, 4)[:, :2].sum(axis=0)
    magnitude = float(np.linalg.norm(reference))
    target_magnitude = float(np.linalg.norm(candidate))
    delta = candidate - reference
    row = dict(
        gen_met=magnitude,
        target_met=target_magnitude,
        gen_met_x=float(reference[0]),
        gen_met_y=float(reference[1]),
        target_met_x=float(candidate[0]),
        target_met_y=float(candidate[1]),
        delta_x=float(delta[0]),
        delta_y=float(delta[1]),
        magnitude_response=None,
        projected_response=None,
        delta_parallel=None,
        delta_perp=None,
        delta_phi=None,
    )
    if magnitude > 1e-9:
        unit = reference / magnitude
        row.update(
            magnitude_response=target_magnitude / magnitude,
            projected_response=float(candidate @ unit / magnitude),
            delta_parallel=float(delta @ unit),
            delta_perp=float(unit[0] * delta[1] - unit[1] * delta[0]),
        )
        if target_magnitude > 1e-9:
            row["delta_phi"] = float(np.arctan2(reference[0] * candidate[1] - reference[1] * candidate[0], reference @ candidate))
    return row


def analyze_event(result, event_index, config):
    gen_particles = [p for p in result["gen"] if abs(p.pdg_id) not in (12, 14, 16)]
    gen_p4 = np.array([p.p4 for p in gen_particles]).reshape(-1, 4)
    mask_pt, mask_eta, _ = kinematics(gen_p4)
    gen_mask = (mask_pt > config.particle_pt_min) & (np.abs(mask_eta) <= config.particle_eta_max + 1e-12)
    target_rows = np.flatnonzero(result["target_id"])
    target_rows = target_rows[~np.isin(result["ytarget"]["pid"][target_rows], (12, 14, 16))]
    target_p4 = result["target_p4"][target_rows]
    target_pt, target_eta, _ = kinematics(target_p4)
    target_mask = (target_pt > config.particle_pt_min) & (np.abs(target_eta) <= config.particle_eta_max + 1e-12)
    gen = fiducial(gen_p4, config)
    target = fiducial(target_p4, config)
    gen_jets = cluster_jets(gen, config.genjet_pt_min, config)
    target_jets = cluster_jets(target, config.targetjet_pt_min, config)
    gp, ge, gf = kinematics(gen_jets)
    tp, te, tf = kinematics(target_jets)
    ig, it = _match_jets_event(ge, gf, te, tf, config.match_dr)
    pairs = []
    for g, t in zip(ig, it):
        dphi = float(np.arctan2(np.sin(tf[t] - gf[g]), np.cos(tf[t] - gf[g])))
        pairs.append(
            dict(
                event_index=event_index,
                event_id=list(result["event_id"]),
                gen_index=int(g),
                target_index=int(t),
                gen_pt=float(gp[g]),
                gen_eta=float(ge[g]),
                target_pt=float(tp[t]),
                target_eta=float(te[t]),
                response=float(tp[t] / gp[g]),
                dr=float(np.hypot(te[t] - ge[g], dphi)),
            )
        )
    met = dict(event_index=event_index, event_id=list(result["event_id"]), **met_quantities(gen, target))
    met["gen_sum_pt"] = float(kinematics(gen)[0].sum())
    met["target_sum_pt"] = float(kinematics(target)[0].sum())
    return dict(
        event_id=list(result["event_id"]),
        event_index=event_index,
        gen_particles=len(gen),
        target_particles=len(target),
        gen_particle_p4=gen.tolist(),
        target_particle_p4=target.tolist(),
        gen_pid=[abs(p.pdg_id) for p, keep in zip(gen_particles, gen_mask) if keep],
        target_pid=[int(result["ytarget"]["pid"][row]) for row, keep in zip(target_rows, target_mask) if keep],
        gen_jets=[dict(pt=float(p), eta=float(e), matched=i in set(ig.tolist())) for i, (p, e) in enumerate(zip(gp, ge))],
        target_jets=[dict(pt=float(p), eta=float(e), matched=i in set(it.tolist())) for i, (p, e) in enumerate(zip(tp, te))],
        jet_matches=pairs,
        met=met,
    )


def robust_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite response")
    if not len(values):
        return dict(n=0, median=None, q16=None, q84=None, sigma68=None, relative_sigma68=None)
    low, median, high = np.quantile(values, [0.16, 0.5, 0.84])
    width = (high - low) / 2
    return dict(
        n=len(values),
        median=float(median),
        q16=float(low),
        q84=float(high),
        sigma68=float(width),
        relative_sigma68=float(width / abs(median)) if abs(median) > 1e-12 else None,
    )


def bootstrap_stats(values, groups, replicas=200, seed=12345):
    result = robust_stats(values)
    if result["n"] < 2 or replicas == 0:
        result["intervals"] = {}
        return result
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chunks = [values[groups == g] for g in unique]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("median", "sigma68", "relative_sigma68")}
    for _ in range(replicas):
        selected = rng.integers(0, len(chunks), size=len(chunks))
        stat = robust_stats(np.concatenate([chunks[i] for i in selected]))
        for key in samples:
            if stat[key] is not None:
                samples[key].append(stat[key])
    result["intervals"] = {key: [float(v) for v in np.quantile(sample, [0.16, 0.84])] for key, sample in samples.items() if sample}
    result["independent_events"] = len(unique)
    return result


def binned_stats(rows, xfield, valuefield, edges, replicas=200, min_count=20, absolute_x=False):
    output = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [r for r in rows if r[valuefield] is not None and low <= (abs(r[xfield]) if absolute_x else r[xfield]) < high]
        stat = bootstrap_stats([r[valuefield] for r in selected], [r["event_index"] for r in selected], replicas, 12345 + index)
        output.append(dict(low=float(low), high=float(high), plot=stat["n"] >= min_count, **stat))
    return output
