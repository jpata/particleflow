#!/usr/bin/env python3
"""Validate postprocessed detector Parquet files.

Run every check, write validation_report.json and plots, and exit 1 if a check fails.
Checks without the required detector data are marked SKIP.
"""

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import awkward as ak
import matplotlib

# Headless CLI/CI runs need a non-interactive backend; but in a notebook pyplot is
# already live with the inline backend by the time this module is imported, and calling
# use("Agg") at that point would clobber it (silently disabling inline rendering for the
# rest of the session). Force Agg only when no backend is active yet.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mlpf.conf import EDM4HEP
from mlpf.data.target_building import visible_energy_deposit, visible_energy_fraction

# ---------------------------------------------------------------------------
# feature layout (see mlpf/conf.py)
# ---------------------------------------------------------------------------
(
    PID,
    CHARGE,
    PT,
    ETA,
    SIN_PHI,
    COS_PHI,
    ENERGY,
    ISPU,
    GEN_STATUS,
    SIM_STATUS,
    GP_TO_TRACK,
    GP_TO_CLUSTER,
    JET_IDX,
    PN,
) = range(14)

# X feature indices
X_TRACK_PT = 1
X_CLUSTER_ENERGY = 5
X_CLUSTER_POSITION = slice(6, 9)
X_CLUSTER_ENERGY_CHERENKOV = 10  # IDEA: common energy_ecal slot
X_CLUSTER_ENERGY_SCINTILLATION = 11  # IDEA: common energy_hcal slot
X_CLUSTER_ENERGY_OTHER = 12
X_HIT_ENERGY = 5
X_HIT_POS_XY = slice(6, 8)
X_ELEMTYPE = 0

HIT_FIELDS = ["X_hit_tracker", "X_hit_calo", "ytarget_hit_tracker", "ytarget_hit_calo"]
PF_FIELDS = ["X_track", "X_cluster", "ytarget_track", "ytarget_cluster"]
YCAND_FIELDS = ["ycand_track", "ycand_cluster"]
REQUIRED_FIELDS = HIT_FIELDS + ["genmet", "genjet", "targetjet"]
ALL_FIELDS = HIT_FIELDS + PF_FIELDS + YCAND_FIELDS + ["genmet", "genjet", "targetjet"]

X_Y_PAIRS = [
    ("X_track", "ytarget_track"),
    ("X_cluster", "ytarget_cluster"),
    ("X_hit_tracker", "ytarget_hit_tracker"),
    ("X_hit_calo", "ytarget_hit_calo"),
]
TARGET_FIELDS = [
    "ytarget_track",
    "ytarget_cluster",
    "ytarget_hit_tracker",
    "ytarget_hit_calo",
]

# PID classes used by the candidate/classifier mapping (postprocessing.py)
ALLOWED_PIDS = {0, 11, 13, 22, 130, 211}
EXPECTED_PIDS = {11, 13, 22, 130, 211}  # e, mu, photon, K0L, charged hadron

# calibrated thresholds (measured on the CLIC 100-event test parquet)
THRESHOLDS = {
    "h3_median": (0.8, 1.2),  # median pT_fit / pT_truth
    "h3_frac": (0.5, 2.0, 0.85),  # fraction of fits within [lo, hi] must be >= this
    "h4_median": (0.7, 1.3),  # median E_sum / E_truth
    "h4_frac": (0.5, 2.0, 0.85),
    "p3_median": 0.05,  # |E(ycand) - E(ytarget)| / E(ytarget)
    "p3_p90": 0.15,
    "g3_median": 0.10,  # |E(ytarget) - (E(genjet) + genmet)| / (E(genjet) + genmet)
    "g3_p90": 0.30,
    "t4_median": (0.5, 2.0),
    "t4_frac": 0.80,
    # Direction diagnostics on native element/target rows, without rematching.
    # Clusters need a wider cone because showers, bending and merged targets
    # can legitimately move their direction. These are quality gates, not
    # assertions that every reconstructed element matches truth exactly.
    "t5_alignment": (0.1, 0.95),  # track delta-R cut, minimum passing fraction
    "t6_alignment": (0.3, 0.80),  # cluster delta-R cut, minimum passing fraction
    "min_fits": 100,
    "v2_calo_median": 100.0,  # |mean(calo hits) - cluster|, mm
    "v2_calo_frac": (0.0, 500.0, 0.90),
    "v2_min_reps": 100,
    "v2_min_hits": 5,
    # target-definition gates (PR #490: merge overwrite, calibrated gp_to_hit
    # weights, MIP-retention OR, simulator-status sanity). Values are
    # calibrated on the CLIC/CLD 100-event test parquets.
    "r1_median": (0.9, 1.1),  # median target/gen jet pT response
    "r1_frac_above_1": 0.02,  # >=2% of matched jets must have response > 1.0
    "r1_min_jets": 50,
    "r2_median": (0.3, 1.1),  # median gp_to_cluster / E_gen per target rep
    "r2_frac_range": (0.0, 1.5, 0.95),  # >=95% of deposit fractions in [0, 1.5]
    "r2_p99": 3.0,  # 99th percentile of deposit fractions
    "r2_min_measurements": 100,
    "m1_visible_energy_fraction": visible_energy_fraction,  # single-source: mlpf/data/target_building.py
    "m1_visible_energy_deposit": visible_energy_deposit,  # single-source: mlpf/data/target_building.py
    "m1_min_muons": 10,
    "m1_min_lowfrac_frac": 0.10,  # >=10% of target muons must be MIP-like (frac < 0.1)
    "e1_endpoint_mask": 0x0F000000,  # EDM4hep bits 24-27 (simulator endpoint flags)
    "w1_neutral_frac_warn": 0.50,  # WARN if neutrals dominate absolute-term admissions
    "w1_min_particles": 3,
}

# jet matching radius for the response gate (matches JET_CONFIG.match_dr in mlpf/conf.py)
JET_MATCH_DR = 0.1

# plotting extent used by the 3D PN visualization (default; ColliderML's ODD geometry is
# bigger — HCAL barrel r ≈ 2.5-3 m, endcaps |z| ≈ 4 m — so it gets a wider window)
VIS_RANGE = 2500.0  # mm
VIS_RANGE_COLLIDERML = 4500.0  # mm


def _deltaphi(phi1: float, phi2: float) -> float:
    """Signed difference of two azimuthal angles in [-pi, pi]."""
    d = phi1 - phi2
    return np.arctan2(np.sin(d), np.cos(d))


def _has_pid_conflict(pn, pid):
    """Detect multiple positive PIDs per positive PN without rescanning all hits per PN."""
    mask = (pn > 0) & (pid > 0)
    pairs = np.unique(np.column_stack((pn[mask], pid[mask])), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return bool(np.any(counts > 1))


def fit_circle(x, y, max_iters=50, threshold=10.0):
    """
    Fit a circle through points (x, y) constrained to pass through the origin
    using RANSAC. Returns the radius R.
    """
    best_inliers_count = 0
    best_params = None

    if len(x) < 3:
        return None

    for _ in range(max_iters):
        idx = np.random.choice(len(x), 2, replace=False)
        xs, ys = x[idx], y[idx]

        z = xs**2 + ys**2
        A = np.stack([2 * xs, 2 * ys], axis=1)
        try:
            params = np.linalg.solve(A, z)
            xc, yc = params
            R = np.sqrt(xc**2 + yc**2)

            distances = np.abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R)
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count

                A_in = np.stack([2 * x[inliers], 2 * y[inliers]], axis=1)
                z_in = x[inliers] ** 2 + y[inliers] ** 2
                best_params, _, _, _ = np.linalg.lstsq(A_in, z_in, rcond=None)
        except Exception:
            continue

    if best_params is not None:
        xc, yc = best_params
        return np.sqrt(xc**2 + yc**2)
    return None


@dataclass
class GateResult:
    gate_id: str
    category: str
    title: str
    expectation: str
    observed: str
    status: str
    plot: str = ""


class ParquetValidator:
    """Validates one parquet file and writes one plot per gate.

    By default every gate plot is saved as a PNG into `plots_dir` (and a
    validation_report.json alongside it). For notebook use, pass `show_plots=True`
    to render each plot inline via plt.show() instead, and `save_plots=False` to
    skip all disk output; the gate results stay available in `self.gates` and
    `self.summary()`."""

    def __init__(self, input_path, detector, max_events, plots_dir, save_plots=True, show_plots=False):
        self.input_path = Path(input_path)
        self.detector = detector
        # The B-field is used only by gate tracker_hit_momentum (tracker-hit circle fits, pT = 0.0003*B*r).
        self.bfield = EDM4HEP.DETECTORS[detector].b_field
        self.has_configured_hits = bool(EDM4HEP.DETECTORS[detector].hit_collections)
        self.vis_range = VIS_RANGE_COLLIDERML if detector.startswith("colliderml") else VIS_RANGE
        self.max_events = max_events
        self.plots_dir = Path(plots_dir)
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.data = None
        self.nev = 0
        self.nev_used = 0
        self.gates = []
        self.alignment_metrics = {}

    # ------------------------------------------------------------- utilities
    def add_gate(self, gate_id, category, title, expectation, observed, status):
        gate = GateResult(gate_id, category, title, expectation, observed, status)
        self.gates.append(gate)
        return gate

    def finish_plot(self, fig, gate):
        colors = {
            "PASS": "#d4edda",
            "FAIL": "#f8d7da",
            "WARN": "#fff3cd",
            "SKIP": "#e2e3e5",
        }
        lines = [gate.title, f"Expected: {gate.expectation}", f"Observed: {gate.observed}", f"Status: {gate.status}"]
        text = "\n".join(textwrap.fill(line, width=110) for line in lines)
        footer_height = 0.18 * len(text.splitlines()) + 0.3
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + footer_height)
        fig.tight_layout(rect=[0, footer_height / (height + footer_height), 1, 1])
        fig.text(
            0.01,
            0.01,
            text,
            transform=fig.transFigure,
            fontsize=9,
            va="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor=colors.get(gate.status, "#ffffff"),
                alpha=0.85,
            ),
        )
        fname = f"{gate.gate_id}.png"
        if self.save_plots:
            fig.savefig(self.plots_dir / fname, dpi=120, bbox_inches="tight")
            gate.plot = fname
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def text_plot(self, gate, lines):
        fig = plt.figure(figsize=(9, 4))
        fig.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=11)
        self.finish_plot(fig, gate)

    def bar_plot(self, gate, labels, values, ylabel, colors=None, hline=None):
        fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(labels) + 3.0), 5))
        ax.bar(labels, values, color=colors or "#4c72b0")
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="black", ls="--", lw=1)
        self.finish_plot(fig, gate)

    def hist_plot(self, gate, data, xlabel, bins=50, shade=None, vlines=(), log=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        if shade is not None:
            ax.axvspan(shade[0], shade[1], color="green", alpha=0.12)
        arr = np.asarray(data, dtype=float)
        # Degenerate inputs directly produce collapsed or missing auto ranges: an array where
        # every value is identical (e.g. jet-response arrays that are exactly 1.0 on
        # colliderml's synthetic test data), or an empty one (colliderml's tracker_hit_momentum circle-fit gate,
        # which has no tracker-hit targets to fit). Give such data an explicit bin range so the
        # gate still gets a readable plot.
        if arr.size == 0:
            ax.bar([0], [0])
        elif float(np.ptp(arr)) < 1e-12:
            v = float(arr[0])
            lo, hi = v - 0.5, v + 0.5
            ax.hist(arr, bins=np.linspace(lo, hi, bins + 1), alpha=0.75, color="#4c72b0", log=log)
        else:
            ax.hist(arr, bins=bins, alpha=0.75, color="#4c72b0", log=log)
        for x, c in vlines:
            ax.axvline(x, color=c, ls="--", lw=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        self.finish_plot(fig, gate)

    def ev(self, field, iev):
        """Event-level numpy view of a var*var field. Shape is either (N, F) or (0, F) for a
        fully-empty event; scalar arrays (e.g. genmet with one scalar per event) come back as
        1-D and callers pick the scalar directly. NOTE: per-event getitem on option-typed
        columns is slow (~0.3 s per event slice) — gates that need all events should slice the
        column once instead of calling this per event (see gate input_target_counts's ak.num vectorization)."""
        a = np.asarray(ak.to_numpy(self.data[field][iev]))
        if a.ndim == 1 and a.size == 0:
            # zero-length 1-D fields (e.g. X_hit_tracker on an event with no tracker hits) come
            # back without a width; callers that slice columns want (0, 0) rather than a scalar.
            return np.zeros((0, 0), dtype=np.float32)
        return a

    # ------------------------------------------------------------------ load
    def load(self):
        try:
            self.data = ak.from_parquet(self.input_path)
        except Exception as exc:  # noqa: BLE001 - report any load failure as gate file_contents
            gate = self.add_gate(
                "file_contents",
                "schema",
                "File contains events",
                "The file parses and contains at least one event.",
                f"load failed: {exc}",
                "FAIL",
            )
            self.text_plot(
                gate,
                [
                    "Could not load parquet file:",
                    str(self.input_path),
                    str(exc),
                    "Status: FAIL",
                ],
            )
            return False
        # The detector registry says whether a key4hep scenario *configures* hit collections;
        # ColliderML carries its hit tables in the parquet itself, so also treat present
        # hit fields as "has hits". (key4hep behavior unchanged: their hit-level productions
        # have both the registry entry and the fields.)
        if not self.has_configured_hits and "X_hit_calo" in self.data.fields:
            self.has_configured_hits = True
        self.nev = 0
        for f in ALL_FIELDS:
            if f in self.data.fields:
                self.nev = int(len(self.data[f]))
                break
        self.nev_used = self.nev if self.max_events is None else min(self.nev, self.max_events)
        return True

    # ------------------------------------------------------------------ gates
    def gate_schema(self):
        ok = self.nev > 0
        gate = self.add_gate(
            "file_contents",
            "schema",
            "File contains events",
            "The file parses and contains at least one event.",
            f"{self.nev} events",
            "PASS" if ok else "FAIL",
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["events"], [self.nev], color="#2ca02c" if ok else "#d62728")
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_ylabel("# events")
        ax.set_ylim(0, max(self.nev, 1) * 1.2)
        self.finish_plot(fig, gate)
        if not ok:
            return

        # required_fields: required fields
        required = list(REQUIRED_FIELDS)
        pf_present = any(f in self.data.fields for f in PF_FIELDS)
        ycand_present = any(f in self.data.fields for f in YCAND_FIELDS)
        if pf_present:
            required += PF_FIELDS
        if ycand_present:
            required += YCAND_FIELDS
        missing = [f for f in required if f not in self.data.fields]
        gate = self.add_gate(
            "required_fields",
            "schema",
            "Required fields are present",
            "Hit-level fields (and genmet/genjet/targetjet) are always required; "
            "track/cluster fields are required when the file is PF-level; "
            "ycand fields are required when present at all.",
            "missing: " + (", ".join(missing) if missing else "none"),
            "PASS" if not missing else "FAIL",
        )
        self.bar_plot(
            gate,
            required,
            [1 if f in self.data.fields else 0 for f in required],
            "present",
            colors=["#2ca02c" if f in self.data.fields else "#d62728" for f in required],
        )
        if missing:
            return

        # input_target_counts: shape consistency X vs ytarget
        mismatches = {}
        for xf, yf in X_Y_PAIRS:
            mismatches[(xf, yf)] = int(np.count_nonzero(ak.to_numpy(ak.num(self.data[xf], axis=1)) != ak.to_numpy(ak.num(self.data[yf], axis=1))))
        total_mismatch = sum(mismatches.values())
        gate = self.add_gate(
            "input_target_counts",
            "schema",
            "Input and target row counts",
            "Each event must have the same number of elements in X and ytarget " "for every (hit, track, cluster) pair.",
            f"{total_mismatch} events with mismatches",
            "PASS" if total_mismatch == 0 else "FAIL",
        )
        self.bar_plot(
            gate,
            [f"{x} vs {y}" for x, y in X_Y_PAIRS],
            list(mismatches.values()),
            "mismatched events",
            colors=["#d62728" if v else "#2ca02c" for v in mismatches.values()],
        )

        # finite_values: finiteness
        nonfinite = {}
        for f in ALL_FIELDS:
            if f in self.data.fields:
                flat = ak.to_numpy(ak.flatten(self.data[f], axis=None))
                nonfinite[f] = int(np.sum(~np.isfinite(flat)))
        total_nonfinite = sum(nonfinite.values())
        gate = self.add_gate(
            "finite_values",
            "schema",
            "Finite feature values",
            "All stored features and targets must be finite.",
            f"{total_nonfinite} non-finite values",
            "PASS" if total_nonfinite == 0 else "FAIL",
        )
        self.bar_plot(
            gate,
            list(nonfinite.keys()),
            list(nonfinite.values()),
            "non-finite values",
            colors=["#d62728" if v else "#2ca02c" for v in nonfinite.values()],
        )

    def gate_hits(self):
        # particle_labels: target consistency across all categories (noise, reps, PID)
        noise_bad = rep_missing = pid_conflict = 0
        for i in range(self.nev_used):
            # reshape(-1, 14) normalizes fully empty events (e.g. no tracker
            # hits) to (0, 14) so concatenation does not fail on a 1-D empty array
            y = np.concatenate([self.ev(f, i).reshape(-1, 14) for f in TARGET_FIELDS])
            pn = y[:, PN].astype(int)
            pid = y[:, PID].astype(int)
            e = y[:, ENERGY]
            if np.any((pn == 0) & ((pid != 0) | (np.abs(e) > 1e-5))):
                noise_bad += 1
            unique_pn = np.unique(pn[pn > 0])
            reps = set(np.unique(pn[pid > 0]))
            if not all(p in reps for p in unique_pn):
                rep_missing += 1
            pid_conflict += int(_has_pid_conflict(pn, pid))
        bad = noise_bad + rep_missing + pid_conflict
        gate = self.add_gate(
            "particle_labels",
            "hits",
            "Particle labels on inputs and targets",
            "Elements with particle_number=0 must have PID=0 and E=0; every "
            "particle_number>0 must have at least one target row (PID>0); "
            "all target rows of a particle must share the same PID.",
            f"noise={noise_bad}, particles without targets={rep_missing}, pid_conflicts={pid_conflict} " f"bad events of {self.nev_used}",
            "PASS" if bad == 0 else "FAIL",
        )
        self.bar_plot(
            gate,
            ["noise", "particles without targets", "PID conflicts"],
            [noise_bad, rep_missing, pid_conflict],
            "events with violations",
            colors=["#d62728" if v else "#2ca02c" for v in (noise_bad, rep_missing, pid_conflict)],
        )

        if not self.has_configured_hits:
            for gate_id, title, detail in [
                (
                    "hit_counts",
                    "Tracker and calorimeter hit counts",
                    "This detector scenario intentionally stores track/cluster inputs without detector hits.",
                ),
                (
                    "tracker_hit_momentum",
                    "Momentum from tracker hits",
                    "Tracker hits are not part of this dataset contract.",
                ),
                (
                    "calorimeter_hit_energy",
                    "Energy from calorimeter hits",
                    "Calorimeter hits are not part of this dataset contract.",
                ),
            ]:
                gate = self.add_gate(
                    gate_id,
                    "hits",
                    title,
                    "Applicable only to hit-level datasets.",
                    detail,
                    "SKIP",
                )
                self.text_plot(gate, [detail])
            return

        # hit_counts: hit multiplicity (informational)
        # reshape also handles intentionally empty hit collections, whose
        # inner dimension is not retained by Awkward/Parquet.
        n_hit_features = len(EDM4HEP.HitFeatures.get_names())

        def _norm_empty(a):
            # only zero-row events lose their column width in the parquet readback; non-empty
            # hit tables keep their native width (the 15-wide EDM4hep layout, key4hep and ColliderML)
            return a.reshape(-1, n_hit_features) if a.size == 0 else a

        n_trk_hits = np.array([int(np.sum(_norm_empty(self.ev("X_hit_tracker", i))[:, X_ELEMTYPE] != 0)) for i in range(self.nev_used)])
        n_calo_hits = np.array([int(np.sum(_norm_empty(self.ev("X_hit_calo", i))[:, X_ELEMTYPE] != 0)) for i in range(self.nev_used)])
        ok = (n_trk_hits > 0).all() and (n_calo_hits > 0).all()
        gate = self.add_gate(
            "hit_counts",
            "hits",
            "Tracker and calorimeter hit counts",
            "Every event should contain tracker and calorimeter hits; "
            "typical medians are >100 tracker and >1000 calo hits per event "
            "(detector-dependent).",
            f"tracker hits median={np.median(n_trk_hits):.0f}, calo hits median={np.median(n_calo_hits):.0f}",
            "PASS" if ok else "WARN",
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(n_trk_hits, bins=50, color="#4c72b0")
        ax1.axvline(100, color="orange", ls="--", lw=1.2, label="expect >= 100")
        ax1.set_xlabel("tracker hits / event")
        ax1.set_ylabel("events")
        ax1.legend(fontsize=8)
        ax2.hist(n_calo_hits, bins=50, color="#4c72b0")
        ax2.axvline(1000, color="orange", ls="--", lw=1.2, label="expect >= 1000")
        ax2.set_xlabel("calo hits / event")
        ax2.legend(fontsize=8)
        self.finish_plot(fig, gate)

        # tracker_hit_momentum/calorimeter_hit_energy: hit-level closure metrics from tracker circle fits and calo
        # energy sums. The shared implementation also writes the 4-panel
        # diagnostic plot into the plots directory.
        ratios_pt = np.array([])
        ratios_e = np.array([])
        inclusive_note = ""
        try:
            incl = self._inclusive_hit_validation()
            ratios_pt = incl["track"]["pT_fit"] / incl["track"]["pT_truth"]
            ratios_e = incl["calo"]["E_hit_sum"] / incl["calo"]["E_truth"]
            if len(ratios_pt):
                ratios_pt = ratios_pt[np.isfinite(ratios_pt)]
            if len(ratios_e):
                ratios_e = ratios_e[np.isfinite(ratios_e)]
        except Exception as exc:  # noqa: BLE001
            inclusive_note = f" (inclusive validation error: {exc})"

        gate = self._closure_gate(
            "tracker_hit_momentum",
            "hits",
            "Momentum from tracker hits",
            "Circle fits through tracker hits of a particle should recover its "
            "gen pT: median pT_fit/pT_truth in 0.8-1.2 and >=85% of fits within "
            "[0.5, 2].",
            ratios_pt,
            THRESHOLDS["h3_median"],
            THRESHOLDS["h3_frac"],
            "pT_fit / pT_truth",
        )
        gate.observed += inclusive_note
        self.hist_plot(
            gate,
            ratios_pt,
            "pT_fit / pT_truth",
            shade=(0.5, 2.0),
            vlines=((0.8, "red"), (1.2, "red")),
        )

        gate = self._closure_gate(
            "calorimeter_hit_energy",
            "hits",
            "Energy from calorimeter hits",
            "The sum of calo hit energies of a particle should match its gen "
            "energy: median E_sum/E_truth in 0.7-1.3 and >=85% of sums within "
            "[0.5, 2].",
            ratios_e,
            THRESHOLDS["h4_median"],
            THRESHOLDS["h4_frac"],
            "E_hit_sum / E_truth",
        )
        gate.observed += inclusive_note
        self.hist_plot(
            gate,
            ratios_e,
            "E_hit_sum / E_truth",
            shade=(0.5, 2.0),
            vlines=((0.7, "red"), (1.3, "red")),
        )

    def gate_visualization(self):
        """detector_positions: sanity of the data behind the 3D PN visualization."""
        n_finite = 0
        n_assigned = 0
        in_range_frac = []
        assigned_frac = []
        for i in range(self.nev_used):
            positions, pns = self._event_positions_pn(i)
            if np.all(np.isfinite(positions)):
                n_finite += 1
            if np.any(pns > 0):
                n_assigned += 1
            if len(positions):
                in_range_frac.append(np.mean(np.max(np.abs(positions), axis=1) <= self.vis_range))
                assigned_frac.append(np.mean(pns > 0))
        in_range_frac = np.array(in_range_frac)
        assigned_frac = np.array(assigned_frac)

        status = "PASS"
        if n_finite != self.nev_used or n_assigned != self.nev_used:
            status = "FAIL"
        elif (len(in_range_frac) and in_range_frac.min() < 0.85) or (len(assigned_frac) and assigned_frac.min() < 0.70):
            status = "WARN"

        observed = (
            f"finite={n_finite}/{self.nev_used}, assigned events={n_assigned}/{self.nev_used}, "
            f"elements in range median={np.median(in_range_frac):.3f}, "
            f"assigned fraction median={np.median(assigned_frac):.3f}"
        )
        gate = self.add_gate(
            "detector_positions",
            "visualization",
            "Positions and particle numbers",
            "Finite element positions, at least one particle assignment (PN>0) per "
            f"event, and most elements inside the plotted volume (+/-{self.vis_range:.0f} mm) and "
            "assigned to a particle.",
            observed,
            status,
        )
        self._pn_visualization_plot(gate)

    def _track_display_positions(self, x, mask):
        """Display positions (x, y, z) of track elements, detector-dependent.

        key4hep X carries radiusOfInnermostHit (col 10) and Z0 (col 14). The ColliderML
        clustered view has neither, so tracks are extrapolated from the perigee parameters
        to a fixed display radius.
        """
        if self.detector.startswith("colliderml"):
            R_DISPLAY = 300.0  # mm, fixed display radius inside the tracker volume
            theta = np.clip(x[mask, 8], 1e-3, np.pi - 1e-3)
            z = np.clip(x[mask, 7] + R_DISPLAY / np.tan(theta), -self.vis_range, self.vis_range)
            return np.stack([R_DISPLAY * x[mask, 4], R_DISPLAY * x[mask, 3], z], axis=1)
        # track position: innermost radius (10), sin_phi (3), cos_phi (4), Z0 (14)
        return np.stack([x[mask, 10] * x[mask, 4], x[mask, 10] * x[mask, 3], x[mask, 14]], axis=1)

    def _event_positions_pn(self, iev):
        """Concatenated element positions (x, y, z) and particle numbers for one event."""
        positions = []
        pns = []
        for f in ["X_hit_tracker", "X_hit_calo", "X_cluster"]:
            if f not in self.data.fields:
                continue
            x = self.ev(f, iev)
            if len(x) == 0:
                continue
            y = self.ev(f.replace("X_", "ytarget_"), iev)
            mask = x[:, X_ELEMTYPE] != 0
            positions.append(x[mask, 6:9])
            pns.append(y[mask, PN])
        if "X_track" in self.data.fields:
            x = self.ev("X_track", iev)
            y = self.ev("ytarget_track", iev)
            if len(x):
                mask = x[:, X_ELEMTYPE] != 0
                positions.append(self._track_display_positions(x, mask))
                pns.append(y[mask, PN])
        positions = np.concatenate(positions) if positions else np.empty((0, 3))
        pns = np.concatenate(pns) if pns else np.empty(0)
        return positions, pns

    def _pn_visualization_plot(self, gate):
        """3D visualization of hits, tracks and clusters colored by particle number."""
        iev = 0
        pos_trk = pn_trk = pos_cl = pn_cl = None

        # reshape only when the event is empty: the parquet readback drops the column width
        # for zero-row events, while non-empty tables keep their native width (the 15-wide
        # EDM4hep hit layout).
        def _hit_ev_pair(fx, fy):
            x, y = self.ev(fx, iev), self.ev(fy, iev)
            if x.size == 0:
                return (
                    np.zeros((0, len(EDM4HEP.HitFeatures.get_names())), dtype=np.float32),
                    np.zeros((0, 14), dtype=np.float32),
                )
            return x, y

        x, y = _hit_ev_pair("X_hit_tracker", "ytarget_hit_tracker")
        mask = x[:, X_ELEMTYPE] != 0
        pos_trk_h, pn_trk_h = x[mask, 6:9], y[mask, PN]

        x, y = _hit_ev_pair("X_hit_calo", "ytarget_hit_calo")
        mask = x[:, X_ELEMTYPE] != 0
        pos_calo_h, pn_calo_h = x[mask, 6:9], y[mask, PN]

        if "X_track" in self.data.fields and "X_cluster" in self.data.fields:
            x = self.ev("X_track", iev)
            y = self.ev("ytarget_track", iev)
            mask = x[:, X_ELEMTYPE] != 0
            pos_trk = self._track_display_positions(x, mask)
            pn_trk = y[mask, PN]

            x = self.ev("X_cluster", iev)
            y = self.ev("ytarget_cluster", iev)
            mask = x[:, X_ELEMTYPE] != 0
            pos_cl, pn_cl = x[mask, 6:9], y[mask, PN]

        all_pns = np.concatenate([pn_trk_h, pn_calo_h] + ([pn_trk, pn_cl] if pn_trk is not None else []))
        vmin, vmax = np.min(all_pns), np.max(all_pns)
        cmap = plt.get_cmap("tab20")

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(
            pos_trk_h[:, 0],
            pos_trk_h[:, 1],
            pos_trk_h[:, 2],
            c=pn_trk_h,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=2,
            label="Tracker Hits",
            alpha=0.5,
        )
        ax1.scatter(
            pos_calo_h[:, 0],
            pos_calo_h[:, 1],
            pos_calo_h[:, 2],
            c=pn_calo_h,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=15,
            marker="s",
            label="Calo Hits",
            alpha=0.7,
        )
        ax1.set_xlabel("X [mm]")
        ax1.set_ylabel("Y [mm]")
        ax1.set_zlabel("Z [mm]")
        ax1.set_title(f"Hits by particle number (Event {iev})")
        ax1.view_init(elev=20, azim=45)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            pos_trk_h[:, 0],
            pos_trk_h[:, 1],
            pos_trk_h[:, 2],
            c=pn_trk_h,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=1,
            alpha=0.03,
        )
        ax2.scatter(
            pos_calo_h[:, 0],
            pos_calo_h[:, 1],
            pos_calo_h[:, 2],
            c=pn_calo_h,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=2,
            marker="s",
            alpha=0.03,
        )
        if pos_trk is not None:
            ax2.scatter(
                pos_trk[:, 0],
                pos_trk[:, 1],
                pos_trk[:, 2],
                c=pn_trk,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=40,
                marker="o",
                edgecolors="black",
                label="Tracks",
                alpha=1.0,
            )
            ax2.scatter(
                pos_cl[:, 0],
                pos_cl[:, 1],
                pos_cl[:, 2],
                c=pn_cl,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=60,
                marker="D",
                edgecolors="black",
                label="Clusters",
                alpha=1.0,
            )
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")
        ax2.set_zlabel("Z [mm]")
        ax2.set_title(f"Tracks and clusters by particle number (Event {iev})")
        ax2.view_init(elev=20, azim=45)

        for ax in [ax1, ax2]:
            ax.set_xlim(-self.vis_range, self.vis_range)
            ax.set_ylim(-self.vis_range, self.vis_range)
            ax.set_zlim(-self.vis_range, self.vis_range)

        self.finish_plot(fig, gate)

    def gate_hit_rep_consistency(self):
        """hit_cluster_positions: mean calo-hit position of a particle matches its cluster position."""
        if not self.has_configured_hits:
            detail = "This detector scenario intentionally stores track/cluster inputs without calorimeter hits."
            gate = self.add_gate(
                "hit_cluster_positions",
                "visualization",
                "Calorimeter hit and cluster positions",
                "Applicable only to datasets containing calorimeter hits.",
                detail,
                "SKIP",
            )
            self.text_plot(gate, [detail])
            return
        min_hits = THRESHOLDS["v2_min_hits"]
        d_clu = []
        for i in range(self.nev_used):
            cluster_pos = self._cluster_rep_positions(i)
            x = self.ev("X_hit_calo", i)
            y = self.ev("ytarget_hit_calo", i)
            if len(x) == 0 or len(y) == 0:
                continue
            mask = x[:, X_ELEMTYPE] != 0
            pns = y[mask, PN].astype(int)
            pos = x[mask, 6:9]
            for p in np.unique(pns):
                if p not in cluster_pos:
                    continue
                h = pos[pns == p]
                if len(h) >= min_hits:
                    d_clu.append(np.linalg.norm(h.mean(axis=0) - cluster_pos[p]))

        d_clu = np.array(d_clu)
        c_med, (_, c_hi, c_frac) = (
            THRESHOLDS["v2_calo_median"],
            THRESHOLDS["v2_calo_frac"],
        )
        min_reps = THRESHOLDS["v2_min_reps"]

        if len(d_clu) < min_reps:
            status = "WARN"
        else:
            status = "PASS" if (np.median(d_clu) < c_med and np.mean(d_clu < c_hi) >= c_frac) else "FAIL"

        observed = f"calo: n={len(d_clu)} median={np.median(d_clu):.1f} mm, " f"frac<{c_hi:.0f} mm={np.mean(d_clu < c_hi):.3f}"
        gate = self.add_gate(
            "hit_cluster_positions",
            "visualization",
            "Calorimeter hit and cluster positions",
            "The mean position of a particle's calorimeter hits should match "
            f"its cluster centroid: median < {c_med:.0f} mm with >={c_frac:.2f} "
            f"of particles within {c_hi:.0f} mm.",
            observed,
            status,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axvline(c_hi, color="red", ls="--", lw=1.2, label=f"expect >= {c_frac:.2f} below")
        ax.hist(d_clu, bins=80, color="#4c72b0")
        ax.set_xlabel("|mean(calo hits) - cluster| [mm]")
        ax.set_ylabel("particles")
        ax.legend(fontsize=8)
        self.finish_plot(fig, gate)

    def _cluster_rep_positions(self, iev):
        """Map particle_number -> cluster position for cluster target rows."""
        cluster_pos = {}
        x = self.ev("X_cluster", iev)
        y = self.ev("ytarget_cluster", iev)
        if len(x) == 0 or len(y) == 0:
            return cluster_pos
        mask = x[:, X_ELEMTYPE] != 0
        pns = y[mask, PN].astype(int)
        pids = y[mask, PID].astype(int)
        pos = x[mask, 6:9]
        for p, pid, pp in zip(pns, pids, pos):
            if pid > 0:
                cluster_pos[int(p)] = pp
        return cluster_pos

    def gate_tracks_clusters(self):
        if not all(f in self.data.fields for f in PF_FIELDS):
            gate = self.add_gate(
                "track_cluster_counts",
                "tracks/clusters",
                "Tracks and clusters present",
                "PF-level files should contain tracks and clusters.",
                "track/cluster fields absent",
                "SKIP",
            )
            self.text_plot(
                gate,
                [
                    "Track/cluster fields not present in this file.",
                    "Track and cluster checks were skipped.",
                ],
            )
            return

        n_trk = np.array([int(ak.sum(self.data["X_track"][i][:, X_ELEMTYPE] != 0)) for i in range(self.nev_used)])
        n_cl = np.array([int(ak.sum(self.data["X_cluster"][i][:, X_ELEMTYPE] != 0)) for i in range(self.nev_used)])
        no_trk = int(np.sum(n_trk == 0))
        no_cl = int(np.sum(n_cl == 0))
        gate = self.add_gate(
            "track_cluster_counts",
            "tracks/clusters",
            "Tracks and clusters present",
            "Every event should contain at least one track and one cluster.",
            f"{no_trk} events without tracks, {no_cl} without clusters",
            "PASS" if no_trk + no_cl == 0 else "WARN",
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(n_trk, bins=50, color="#4c72b0")
        ax1.set_xlabel("tracks / event")
        ax1.axvline(1, color="orange", ls="--", lw=1.2)
        ax2.hist(n_cl, bins=50, color="#4c72b0")
        ax2.set_xlabel("clusters / event")
        ax2.axvline(1, color="orange", ls="--", lw=1.2)
        self.finish_plot(fig, gate)

        # particle_targets: every PN represented by a track/cluster with positive kinematics
        unrepped_events = nonpositive_events = 0
        for i in range(self.nev_used):
            pn_all, pid_all, pt_all, e_all = [], [], [], []
            for f in ["ytarget_track", "ytarget_cluster"]:
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                pn_all.append(y[:, PN])
                pid_all.append(y[:, PID])
                pt_all.append(y[:, PT])
                e_all.append(y[:, ENERGY])
            pn = np.concatenate(pn_all).astype(int)
            pid = np.concatenate(pid_all).astype(int)
            pt = np.concatenate(pt_all)
            e = np.concatenate(e_all)
            unique_pn = np.unique(pn[pn > 0])
            reps = set(np.unique(pn[pid > 0]))
            if not all(p in reps for p in unique_pn):
                unrepped_events += 1
            if np.any((pid > 0) & ((pt <= 0) | (e <= 0))):
                nonpositive_events += 1
        bad = unrepped_events + nonpositive_events
        gate = self.add_gate(
            "particle_targets",
            "tracks/clusters",
            "Particles with track or cluster targets",
            "Every truth particle (particle_number>0) must be represented by a "
            "track or cluster, and target rows must have positive pt and energy.",
            f"particles without targets={unrepped_events}, non-positive momentum or energy={nonpositive_events} " f"bad events of {self.nev_used}",
            "PASS" if bad == 0 else "FAIL",
        )
        self.bar_plot(
            gate,
            ["particles without targets", "non-positive momentum or energy"],
            [unrepped_events, nonpositive_events],
            "events with violations",
            colors=["#d62728" if v else "#2ca02c" for v in (unrepped_events, nonpositive_events)],
        )

        # target_charge: charge sanity rules.
        # Note: neutral particles reconstructed as tracks keep gen charge 0 but
        # get a forced charged PID (postprocessing), so track PID/charge need
        # not agree; only check |charge| <= 1 for track reps.
        cluster_charge_bad = track_charge_bad = 0
        for i in range(self.nev_used):
            yc = self.ev("ytarget_cluster", i)
            yt = self.ev("ytarget_track", i)
            if len(yc):
                if np.any((yc[:, PID] > 0) & (yc[:, CHARGE] != 0)):
                    cluster_charge_bad += 1
            if len(yt):
                pid = yt[:, PID].astype(int)
                charge = yt[:, CHARGE]
                rep = pid > 0
                if np.any(rep & (np.abs(charge) > 1)):
                    track_charge_bad += 1
        gate = self.add_gate(
            "target_charge",
            "tracks/clusters",
            "Track and cluster target charges",
            "Cluster target rows must be neutral (charge=0); track " "target rows must have |charge| <= 1.",
            f"cluster charge violations={cluster_charge_bad}, track |q|>1 events={track_charge_bad}",
            "FAIL" if cluster_charge_bad else ("WARN" if track_charge_bad else "PASS"),
        )
        self.bar_plot(
            gate,
            ["cluster charge", "track charge"],
            [cluster_charge_bad, track_charge_bad],
            "events with violations",
            colors=["#d62728" if v else "#2ca02c" for v in (cluster_charge_bad, track_charge_bad)],
        )

        # element_target_momentum: kinematic consistency of reps vs reconstructed elements
        r_pt = []
        r_e = []
        for i in range(self.nev_used):
            x = self.ev("X_track", i)
            y = self.ev("ytarget_track", i)
            if len(x) and len(y):
                m = (x[:, X_ELEMTYPE] != 0) & (y[:, PID] > 0) & (x[:, X_TRACK_PT] > 0)
                r_pt.extend((y[m, PT] / x[m, X_TRACK_PT]).tolist())
            x = self.ev("X_cluster", i)
            y = self.ev("ytarget_cluster", i)
            if len(x) and len(y):
                m = (x[:, X_ELEMTYPE] != 0) & (y[:, PID] > 0) & (x[:, X_CLUSTER_ENERGY] > 0)
                r_e.extend((y[m, ENERGY] / x[m, X_CLUSTER_ENERGY]).tolist())
        r_pt = np.array(r_pt)
        r_e = np.array(r_e)
        lo, hi = THRESHOLDS["t4_median"]
        frac_ok = THRESHOLDS["t4_frac"]
        ok = (
            len(r_pt) > 0
            and len(r_e) > 0
            and lo <= np.median(r_pt) <= hi
            and lo <= np.median(r_e) <= hi
            and np.mean((r_pt >= 0.5) & (r_pt <= 2.0)) >= frac_ok
            and np.mean((r_e >= 0.5) & (r_e <= 2.0)) >= frac_ok
        )
        gate = self.add_gate(
            "element_target_momentum",
            "tracks/clusters",
            "Track momentum and cluster energy",
            "Reconstructed track pt (cluster energy) should be close to the "
            "assigned truth: median ratio within 0.5-2 and >=80% of elements "
            "within [0.5, 2].",
            f"track pt: n={len(r_pt)} median={np.median(r_pt) if len(r_pt) else float('nan'):.3f}; "
            f"cluster E: n={len(r_e)} median={np.median(r_e) if len(r_e) else float('nan'):.3f}",
            "PASS" if ok else "WARN",
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.axvspan(0.5, 2.0, color="green", alpha=0.12)
        ax1.hist(r_pt, bins=50, color="#4c72b0")
        ax1.set_xlabel("Target pT / track pT")
        ax2.axvspan(0.5, 2.0, color="green", alpha=0.12)
        ax2.hist(r_e, bins=50, color="#4c72b0")
        ax2.set_xlabel("Target energy / cluster energy")
        self.finish_plot(fig, gate)

    def gate_element_target_alignment(self):
        """Check native assigned rows, excluding PID=0 (including PN-only marks)."""
        for gate_id, kind, threshold in [
            ("track_target_direction", "track", "t5_alignment"),
            ("cluster_target_direction", "cluster", "t6_alignment"),
        ]:
            xf, yf = f"X_{kind}", f"ytarget_{kind}"
            direction_source = "cluster_position" if kind == "cluster" else "track_momentum"
            direction_label = "Cluster position" if kind == "cluster" else "Track momentum"
            cut, required = THRESHOLDS[threshold]
            if xf not in self.data.fields or yf not in self.data.fields:
                gate = self.add_gate(
                    gate_id,
                    "alignment",
                    f"{direction_label} and target particle directions",
                    "Compare each input with its assigned target.",
                    "fields absent",
                    "SKIP",
                )
                self.text_plot(gate, [gate.observed])
                continue
            rows, malformed = [], 0
            for iev in range(self.nev_used):
                x, y = self.ev(xf, iev), self.ev(yf, iev)
                if len(x) != len(y) or (len(x) and (x.shape[1] < (9 if kind == "cluster" else 5) or y.shape[1] < 14)):
                    malformed += 1
                    continue
                if not len(x):
                    continue
                mask = (x[:, X_ELEMTYPE] != 0) & (y[:, PID] > 0)
                x, y = x[mask], y[mask]
                if kind == "cluster":
                    # Pandora's stored angles differ from the direction from
                    # the origin to the cluster; use its position for ownership.
                    position = x[:, X_CLUSTER_POSITION]
                    radius = np.hypot(position[:, 0], position[:, 1])
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        eta = np.arcsinh(position[:, 2] / radius)
                    phi = np.arctan2(position[:, 1], position[:, 0])
                    input_valid = np.isfinite(position).all(axis=1) & (radius > 0)
                else:
                    eta = x[:, 2]
                    phi = np.arctan2(x[:, 3], x[:, 4])
                    input_valid = np.isfinite(x[:, 2:5]).all(axis=1) & (np.hypot(x[:, 3], x[:, 4]) > 0)
                deta = eta - y[:, ETA]
                dphi = _deltaphi(phi, np.arctan2(y[:, SIN_PHI], y[:, COS_PHI]))
                dr = np.hypot(deta, dphi)
                valid = (
                    np.isfinite(dr)
                    & input_valid
                    & np.isfinite(y[:, ETA : COS_PHI + 1]).all(axis=1)
                    & (np.hypot(y[:, SIN_PHI], y[:, COS_PHI]) > 0)
                )
                rows.append(np.column_stack((dr, valid, y[:, PT], y[:, ISPU])))
            a = np.concatenate(rows) if rows else np.empty((0, 4))
            metrics = {"direction_source": direction_source, "delta_r_cut": cut, "minimum_fraction": required, "malformed_events": malformed}
            groups = {
                "all": np.ones(len(a), dtype=bool),
                "hard_scatter": a[:, 3] == 0,
                "pileup": a[:, 3] == 1,
                "pt_ge_1": a[:, 2] >= 1,
                "pt_ge_5": a[:, 2] >= 5,
            }
            for name, mask in groups.items():
                dr, valid = a[mask, 0], a[mask, 1].astype(bool)
                finite = dr[valid]
                metrics[name] = {
                    "count": len(dr),
                    "invalid": int(np.sum(~valid)),
                    "fraction_aligned": float(np.mean(valid & (dr < cut))) if len(dr) else None,
                    "median_delta_r": float(np.median(finite)) if len(finite) else None,
                    "p90_delta_r": float(np.quantile(finite, 0.9)) if len(finite) else None,
                }
            self.alignment_metrics[kind] = metrics
            total = metrics["all"]
            status = "FAIL" if malformed or (len(a) and total["fraction_aligned"] < required) else ("PASS" if len(a) else "SKIP")
            observed = f"n={len(a)}, malformed events={malformed}, invalid={total['invalid']}"
            if len(a):
                median = total["median_delta_r"] if total["median_delta_r"] is not None else float("nan")
                p90 = total["p90_delta_r"] if total["p90_delta_r"] is not None else float("nan")
                observed += f"; within cut={total['fraction_aligned']:.2%}, median delta-R={median:.4g}, p90={p90:.4g}"
            gate = self.add_gate(
                gate_id,
                "alignment",
                f"{direction_label} and target particle directions",
                f">={required:.0%} of inputs with assigned target particles must have delta-R < {cut}. Directions must be finite and defined.",
                observed,
                status,
            )
            self.hist_plot(
                gate,
                a[a[:, 1].astype(bool), 0],
                f"Delta-R between {direction_label.lower()} direction and assigned target particle",
                vlines=((cut, "red"),),
            )

    def gate_targets(self):
        # target_momentum_agreement: target row kinematics consistent across categories
        inconsistent = 0
        reps_per_pn = []
        for i in range(self.nev_used):
            pn_all, pt_all, e_all = [], [], []
            for f in TARGET_FIELDS:
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                rep = y[:, PID] > 0
                pn_all.append(y[rep, PN])
                pt_all.append(y[rep, PT])
                e_all.append(y[rep, ENERGY])
            pn = np.concatenate(pn_all).astype(int)
            pt = np.concatenate(pt_all)
            e = np.concatenate(e_all)
            for p in np.unique(pn):
                m = pn == p
                reps_per_pn.append(int(np.sum(m)))
                if np.sum(m) > 1:
                    if not (np.allclose(pt[m], pt[m][0]) and np.allclose(e[m], e[m][0])):
                        inconsistent += 1
        gate = self.add_gate(
            "target_momentum_agreement",
            "targets",
            "Particle momentum in each target array",
            "A particle may appear at several levels (track/cluster and hit), "
            "but all its target rows must carry identical kinematics "
            "(pt/energy).",
            f"{inconsistent} particles with inconsistent target rows",
            "PASS" if inconsistent == 0 else "WARN",
        )
        self.hist_plot(gate, np.array(reps_per_pn), "target rows per particle", bins=20)

        # target_jet_indices: jet_idx validity
        bad_events = 0
        for i in range(self.nev_used):
            n_jets = len(self.data["targetjet"][i])
            for f in ["ytarget_track", "ytarget_cluster"]:
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                jidx = y[y[:, PID] > 0, JET_IDX].astype(int)
                if np.any((jidx < -1) | (jidx >= n_jets)):
                    bad_events += 1
                    break
        gate = self.add_gate(
            "target_jet_indices",
            "targets",
            "Target jet indices",
            "Representative jet_idx must be -1 (not in a jet) or within " "[0, n_targetjets).",
            f"{bad_events} events with out-of-range jet_idx of {self.nev_used}",
            "PASS" if bad_events == 0 else "WARN",
        )
        self.bar_plot(
            gate,
            ["events with bad jet_idx"],
            [bad_events],
            "events",
            colors=["#ff7f0e" if bad_events else "#2ca02c"],
        )

    def gate_target_definition(self):
        """target_jet_response/deposited_energy/muon_energy_deposits/simulation_flags/neutral_energy_deposits: target-definition invariants raised in PR #490.

        target_jet_response - the target jet response must be a smear around 1.0, not a hard
             ceiling: a downward-only response was the symptom of the
             merge-overwrite energy loss, which also affects CLIC/CLD in
             principle even though it only fired on MAIA in practice.
        deposited_energy - gp_to_cluster (the visibility-mask input) must be a calibrated
             energy in GeV, so deposit/energy is a true energy fraction.
        muon_energy_deposits - the absolute MIP-deposit term must actually retain muons: on
             ttbar-like samples a healthy fraction of target muons are
             MIP-like (deposited fraction below visible_energy_fraction).
        simulation_flags - every depositing target target row must carry simulator
             endpoint bits (EDM4hep bits 24-27); missing flags indicate
             unsimulated particles (e.g. the MAIA 0x80000000 sample issue).
        neutral_energy_deposits - particles admitted only by the absolute term should not be
             dominated by neutrals, whose energy nothing measured
             (informational; mirrors the open question in PR #490).
        """
        if not all(f in self.data.fields for f in PF_FIELDS):
            gate = self.add_gate(
                "target_jet_response",
                "targets",
                "Track and cluster target arrays",
                "These checks require track and cluster target arrays.",
                "track/cluster fields absent",
                "SKIP",
            )
            self.text_plot(
                gate,
                [
                    "Track/cluster fields not present in this file.",
                    "Jet response, energy deposit and simulation flag checks were skipped.",
                ],
            )
            return

        # target_jet_response: jet response, target jet matched to the nearest gen jet
        responses = []
        for i in range(self.nev_used):
            gj = self.ev("genjet", i)
            tj = self.ev("targetjet", i)
            if len(gj) == 0 or len(tj) == 0:
                continue
            for tjet in tj:
                deta = gj[:, 1] - tjet[1]
                dphi = _deltaphi(gj[:, 2], tjet[2])
                dr = np.sqrt(deta**2 + dphi**2)
                m = int(np.argmin(dr))
                if dr[m] < JET_MATCH_DR:
                    responses.append(tjet[0] / gj[m, 0])
        responses = np.array(responses)
        r1_lo, r1_hi = THRESHOLDS["r1_median"]
        r1_min_above = THRESHOLDS["r1_frac_above_1"]
        r1_min_jets = THRESHOLDS["r1_min_jets"]
        if len(responses) < r1_min_jets:
            status = "WARN"
        else:
            frac_above = float(np.mean(responses > 1.0))
            ok = (r1_lo <= np.median(responses) <= r1_hi) and (frac_above >= r1_min_above)
            status = "PASS" if ok else "FAIL"
        observed = (
            f"matched jets={len(responses)}, "
            f"median={np.median(responses) if len(responses) else float('nan'):.3f}, "
            f"frac>1.0={float(np.mean(responses > 1.0)) if len(responses) else float('nan'):.3f}"
        )
        gate = self.add_gate(
            "target_jet_response",
            "targets",
            "Target jet momentum relative to truth",
            f"Median target/truth jet pT in [{r1_lo}, {r1_hi}]; " f">={r1_min_above:.0%} of matched target jets have more pT than the truth jet.",
            observed,
            status,
        )
        self.hist_plot(gate, responses, "Target jet pT / truth jet pT", vlines=((1.0, "red"),))

        if self.detector == "idea":
            # IDEA's two readout signals measure the same physical shower, so
            # their combined calibrated response is not a deposited-energy
            # fraction and is not expected to be bounded by generator energy.
            chunks = [self.ev("X_cluster", i) for i in range(self.nev_used)]
            xcluster = np.concatenate([x for x in chunks if len(x)]) if any(len(x) for x in chunks) else np.empty((0, 0))
            if len(xcluster):
                energy = xcluster[:, X_CLUSTER_ENERGY]
                cherenkov = xcluster[:, X_CLUSTER_ENERGY_CHERENKOV]
                scintillation = xcluster[:, X_CLUSTER_ENERGY_SCINTILLATION]
                other = xcluster[:, X_CLUSTER_ENERGY_OTHER]
                residual = energy - cherenkov - scintillation
                tolerance = 1e-5 + 1e-5 * np.abs(energy)
                bad_closure = np.abs(residual) > tolerance
                bad_sign = (cherenkov < -1e-7) | (scintillation < -1e-7)
                bad_other = np.abs(other) > tolerance
                ok = not np.any(bad_closure | bad_sign | bad_other)
                observed = (
                    f"clusters={len(energy)}, bad closure={int(np.sum(bad_closure))}, "
                    f"negative components={int(np.sum(bad_sign))}, nonzero energy_other={int(np.sum(bad_other))}, "
                    f"max |E-(C+S)|={float(np.max(np.abs(residual))):.3g} GeV"
                )
            else:
                residual = np.empty(0)
                ok = False
                observed = "no IDEA clusters"
            gate = self.add_gate(
                "deposited_energy",
                "targets",
                "IDEA cluster energy components",
                "For IDEA, energy_ecal is Cherenkov energy and energy_hcal is scintillation energy. "
                "Both must be nonnegative, energy_other must be zero, and cluster energy must equal their sum.",
                observed,
                "PASS" if ok else "FAIL",
            )
            self.hist_plot(
                gate,
                residual,
                "cluster E - Cherenkov E - scintillation E [GeV]",
                bins=80,
                vlines=((0.0, "red"),),
            )
        else:
            # deposited_energy: deposited-energy fraction of target target rows
            fracs = []
            r2_fields = ["ytarget_track", "ytarget_cluster"]
            fracs = []
            for f in r2_fields:
                for i in range(self.nev_used):
                    y = self.ev(f, i)
                    if len(y) == 0:
                        continue
                    rep = y[:, PID] > 0
                    e = y[rep, ENERGY]
                    gc = y[rep, GP_TO_CLUSTER]
                    valid = e > 0
                    fracs.extend((gc[valid] / e[valid]).tolist())
            fracs = np.array(fracs)
            r2_lo, r2_hi = THRESHOLDS["r2_median"]
            flo, fhi, fmin = THRESHOLDS["r2_frac_range"]
            p99_lim = THRESHOLDS["r2_p99"]
            n_frac = len(fracs)
            if n_frac < THRESHOLDS["r2_min_measurements"]:
                status = "WARN"
            else:
                med = float(np.median(fracs))
                in_range = float(np.mean((fracs >= flo) & (fracs <= fhi)))
                p99 = float(np.percentile(fracs, 99))
                ok = (r2_lo <= med <= r2_hi) and (in_range >= fmin) and (p99 < p99_lim)
                status = "PASS" if ok else "FAIL"
            observed = (
                f"n={n_frac}, median={np.median(fracs) if n_frac else float('nan'):.3f}, "
                f"in [{flo}, {fhi}]={float(np.mean((fracs >= flo) & (fracs <= fhi))) if n_frac else float('nan'):.3f}, "
                f"p99={float(np.percentile(fracs, 99)) if n_frac else float('nan'):.2f}"
            )
            gate = self.add_gate(
                "deposited_energy",
                "targets",
                "Calorimeter deposit relative to particle energy",
                "gp_to_cluster (the visibility-mask input) must be an energy in "
                f"GeV: median deposit/energy in [{r2_lo}, {r2_hi}], >={fmin:.0%} "
                f"within [{flo}, {fhi}], and p99 < {p99_lim}.",
                observed,
                status,
            )
            self.hist_plot(
                gate,
                fracs,
                "Calorimeter deposit / truth particle energy",
                bins=80,
                shade=(flo, fhi),
                vlines=((1.0, "red"),),
            )

        # muon_energy_deposits: Muon energy deposits
        frac_lim = THRESHOLDS["m1_visible_energy_fraction"]
        dep_lim = THRESHOLDS["m1_visible_energy_deposit"]
        min_muons = THRESHOLDS["m1_min_muons"]
        min_lowfrac_frac = THRESHOLDS["m1_min_lowfrac_frac"]
        mu_frac, mu_dep = [], []
        for f in ["ytarget_track", "ytarget_cluster"]:
            for i in range(self.nev_used):
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                m = (y[:, PID] == 13) & (y[:, ENERGY] > 0)
                mu_frac.extend((y[m, GP_TO_CLUSTER] / y[m, ENERGY]).tolist())
                mu_dep.extend(y[m, GP_TO_CLUSTER].tolist())
        mu_frac = np.array(mu_frac)
        mu_dep = np.array(mu_dep)
        n_mu = len(mu_frac)
        n_lowfrac = int(np.sum(mu_frac < frac_lim)) if n_mu else 0
        n_abs = int(np.sum((mu_frac < frac_lim) & (mu_dep > dep_lim))) if n_mu else 0
        if n_mu < min_muons:
            status = "WARN"
        elif n_mu and (n_lowfrac / n_mu) >= min_lowfrac_frac:
            status = "PASS"
        else:
            status = "FAIL"
        gate = self.add_gate(
            "muon_energy_deposits",
            "targets",
            "Muon energy deposits",
            f"In ttbar samples, at least {min_lowfrac_frac:.0%} of target muons should have "
            f"deposit/energy < {frac_lim:.0%}. Some should also have deposit > {dep_lim:.1f} GeV.",
            f"muons={n_mu}, frac<{frac_lim:.0%}={n_lowfrac} " f"({100 * n_lowfrac / max(n_mu, 1):.0f}%), " f"with deposit>{dep_lim:.1f} GeV={n_abs}",
            status,
        )
        self.bar_plot(
            gate,
            ["target muons", f"frac<{frac_lim:.0%}", f"deposit>{dep_lim:.1f} GeV"],
            [n_mu, n_lowfrac, n_abs],
            "count",
            colors=["#4c72b0", "#2ca02c", "#ff7f0e"],
        )

        # simulation_flags: simulator endpoint bits on all target target rows
        endpoint_mask = THRESHOLDS["e1_endpoint_mask"]
        n_reps = 0
        n_endpoint_viol = 0
        for f in ["ytarget_track", "ytarget_cluster"]:
            for i in range(self.nev_used):
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                rep = y[:, PID] > 0
                n_reps += int(np.sum(rep))
                if np.any(rep):
                    sim = y[rep, SIM_STATUS].astype(np.int64)
                    n_endpoint_viol += int(np.sum((sim & endpoint_mask) == 0))
        status = "WARN" if n_reps == 0 else ("PASS" if n_endpoint_viol == 0 else "FAIL")
        gate = self.add_gate(
            "simulation_flags",
            "targets",
            "Particle simulation flags",
            "Every particle that deposited energy must have at least one "
            f"EDM4hep simulator endpoint flag set (bits 24-27, mask "
            f"0x{endpoint_mask:08X}); missing flags indicate unsimulated "
            "particles without detector interactions in the targets.",
            f"target rows={n_reps}, without endpoint bits={n_endpoint_viol}",
            status,
        )
        self.bar_plot(
            gate,
            ["target rows", "no endpoint bits"],
            [n_reps, n_endpoint_viol],
            "count",
            colors=[
                "#2ca02c" if n_endpoint_viol == 0 else "#d62728",
                "#d62728" if n_endpoint_viol else "#2ca02c",
            ],
        )

        # neutral_energy_deposits: neutrals among absolute-term-only admissions (informational)
        abs_pids = []
        for f in ["ytarget_track", "ytarget_cluster"]:
            for i in range(self.nev_used):
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                e = y[:, ENERGY]
                gc = y[:, GP_TO_CLUSTER]
                rep = y[:, PID] > 0
                denom = np.where(e > 0, e, 1.0)
                abs_only = rep & (gc / denom < frac_lim) & (gc > dep_lim)
                abs_pids.extend(y[abs_only, PID].tolist())
        abs_pids = np.array(abs_pids)
        n_abs_total = len(abs_pids)
        n_neutral = int(np.sum(np.isin(abs_pids, [22, 130])))
        warn_frac = THRESHOLDS["w1_neutral_frac_warn"]
        min_part = THRESHOLDS["w1_min_particles"]
        if n_abs_total < min_part:
            status = "WARN"
        else:
            neutral_frac = n_neutral / n_abs_total
            status = "WARN" if neutral_frac > warn_frac else "PASS"
        gate = self.add_gate(
            "neutral_energy_deposits",
            "targets",
            "Neutral particles with small energy deposits",
            f"Among particles with deposit/energy < {frac_lim:.0%} and deposit > {dep_lim:.1f} GeV, "
            f"warn if more than {warn_frac:.0%} are neutral.",
            f"particles below the fractional deposit cut={n_abs_total}, neutrals={n_neutral} " f"({100 * n_neutral / max(n_abs_total, 1):.0f}%)",
            status,
        )
        self.bar_plot(
            gate,
            ["small fractional deposits", "neutral particles (22/130)"],
            [n_abs_total, n_neutral],
            "count",
            colors=["#4c72b0", "#ff7f0e" if n_neutral else "#2ca02c"],
        )

    def gate_baseline_pf(self):
        if not all(f in self.data.fields for f in YCAND_FIELDS):
            gate = self.add_gate(
                "baseline_particle_counts",
                "baseline PF",
                "Baseline PF fields present",
                "Files with a baseline PF (ycand_*) should contain both " "ycand_track and ycand_cluster.",
                "ycand fields absent",
                "SKIP",
            )
            self.text_plot(
                gate,
                [
                    "Baseline PF (ycand) fields not present in this file.",
                    "Baseline particle checks were skipped.",
                ],
            )
            return

        # baseline_particle_counts: counts
        mismatch = 0
        for i in range(self.nev_used):
            for xf, yf in [("X_track", "ycand_track"), ("X_cluster", "ycand_cluster")]:
                if len(self.data[xf][i]) != len(self.data[yf][i]):
                    mismatch += 1
                    break
        gate = self.add_gate(
            "baseline_particle_counts",
            "baseline PF",
            "Baseline PF element counts",
            "Each reconstructed element must have exactly one baseline PF entry " "(len(ycand) == len(X)).",
            f"{mismatch} events with count mismatches of {self.nev_used}",
            "PASS" if mismatch == 0 else "FAIL",
        )
        self.bar_plot(
            gate,
            ["events with mismatches"],
            [mismatch],
            "events",
            colors=["#d62728" if mismatch else "#2ca02c"],
        )

        # baseline_particle_labels: PID validity
        pids = []
        bad_pid = bad_energy = 0
        for i in range(self.nev_used):
            for xf, yf in [("X_track", "ycand_track"), ("X_cluster", "ycand_cluster")]:
                x = self.ev(xf, i)
                y = self.ev(yf, i)
                if len(x) == 0 or len(y) == 0:
                    continue
                m = x[:, X_ELEMTYPE] != 0
                pid = y[m, PID].astype(int)
                e = y[m, ENERGY]
                pids.extend(pid.tolist())
                if np.any(~np.isin(pid, list(ALLOWED_PIDS))):
                    bad_pid += 1
                if np.any((pid == 0) & (np.abs(e) > 1e-5)):
                    bad_energy += 1
        gate = self.add_gate(
            "baseline_particle_labels",
            "baseline PF",
            "Baseline particle types",
            "Baseline PIDs must belong to the candidate classes " f"{sorted(ALLOWED_PIDS)} and PID=0 elements must have zero energy.",
            f"bad PID events={bad_pid}, zero-PID non-zero-E events={bad_energy}",
            "FAIL" if (bad_pid or bad_energy) else "PASS",
        )
        labels = sorted(set(pids))
        counts = [pids.count(p) for p in labels]
        self.bar_plot(
            gate,
            [str(p) for p in labels],
            counts,
            "elements",
            colors=["#d62728" if p not in ALLOWED_PIDS else "#4c72b0" for p in labels],
        )

        # baseline_particle_energy: baseline PF vs target closure
        rel = []
        for i in range(self.nev_used):
            e_target = float(ak.sum(self.data["ytarget_track"][i][:, ENERGY])) + float(ak.sum(self.data["ytarget_cluster"][i][:, ENERGY]))
            e_ycand = float(ak.sum(self.data["ycand_track"][i][:, ENERGY])) + float(ak.sum(self.data["ycand_cluster"][i][:, ENERGY]))
            if e_target > 0:
                rel.append(abs(e_ycand - e_target) / e_target)
        rel = np.array(rel)
        ok = len(rel) > 0 and np.median(rel) < THRESHOLDS["p3_median"] and np.percentile(rel, 90) < THRESHOLDS["p3_p90"]
        gate = self.add_gate(
            "baseline_particle_energy",
            "baseline PF",
            "Baseline energy relative to target energy",
            "The baseline PF should reconstruct the target energy: per-event "
            f"|E(ycand)-E(ytarget)|/E(ytarget) median < {THRESHOLDS['p3_median']} "
            f"and 90th percentile < {THRESHOLDS['p3_p90']}.",
            f"median={np.median(rel):.4f}, p90={np.percentile(rel, 90):.4f}",
            "PASS" if ok else "FAIL",
        )
        self.hist_plot(
            gate,
            rel,
            "|E(ycand) - E(ytarget)| / E(ytarget)",
            vlines=(
                (THRESHOLDS["p3_median"], "red"),
                (THRESHOLDS["p3_p90"], "darkred"),
            ),
        )

    def gate_truth(self):
        gm = np.asarray(ak.to_numpy(self.data["genmet"]))
        finite_ok = bool(np.all(np.isfinite(gm)) and np.all(gm >= 0))
        gate = self.add_gate(
            "truth_missing_momentum",
            "ground truth",
            "Truth missing transverse momentum",
            "genmet is derived from status-1 pythia particles and must be a " "finite, non-negative scalar.",
            f"min={gm.min():.2f} GeV, median={np.median(gm):.2f} GeV",
            "PASS" if finite_ok else "FAIL",
        )
        self.hist_plot(gate, gm, "genmet [GeV]", vlines=((0.0, "red"),))

        n_jet = np.array([int(ak.sum(self.data["genjet"][i][:, 3] > 0)) for i in range(self.nev_used)])
        frac_with_jet = np.mean(n_jet > 0)
        gate = self.add_gate(
            "truth_jet_counts",
            "ground truth",
            "Truth jet counts",
            "Every event should contain at least one genjet (sample-dependent; " "expect >= 99% for ttbar-like samples).",
            f"{int(np.sum(n_jet > 0))}/{self.nev_used} events with genjets",
            "PASS" if frac_with_jet >= 0.99 else "WARN",
        )
        self.bar_plot(
            gate,
            ["with genjet", "without genjet"],
            [int(np.sum(n_jet > 0)), int(np.sum(n_jet == 0))],
            "events",
            colors=["#2ca02c", "#ff7f0e"],
        )

        # target_truth_energy: target vs truth closure (genjet energy + genmet)
        rel = []
        for i in range(self.nev_used):
            e_target = float(ak.sum(self.data["ytarget_track"][i][:, ENERGY])) + float(ak.sum(self.data["ytarget_cluster"][i][:, ENERGY]))
            e_truth = float(ak.sum(self.data["genjet"][i][:, 3])) + float(self.data["genmet"][i])
            if e_truth > 0:
                rel.append(abs(e_target - e_truth) / e_truth)
        rel = np.array(rel)
        ok = len(rel) > 0 and np.median(rel) < THRESHOLDS["g3_median"] and np.percentile(rel, 90) < THRESHOLDS["g3_p90"]
        if self.detector == "idea":
            status = "WARN"
            expectation = (
                "Informational only for IDEA: selected gen jets plus transverse MET are not a complete "
                "visible-energy reference for an e+e- event. A future parquet field must store the full "
                "visible generator energy before this can be a closure gate."
            )
        else:
            status = "PASS" if ok else "FAIL"
            expectation = (
                "The target should reproduce the pythia truth: per-event "
                f"|E(ytarget)-(E(genjet)+genmet)|/(E(genjet)+genmet) median < "
                f"{THRESHOLDS['g3_median']} and 90th percentile < {THRESHOLDS['g3_p90']}."
            )
        gate = self.add_gate(
            "target_truth_energy",
            "ground truth",
            "Target energy relative to truth energy",
            expectation,
            f"median={np.median(rel):.4f}, p90={np.percentile(rel, 90):.4f}",
            status,
        )
        self.hist_plot(
            gate,
            rel,
            "|E(ytarget) - (E(genjet)+genmet)| / (E(genjet)+genmet)",
            vlines=(
                (THRESHOLDS["g3_median"], "red"),
                (THRESHOLDS["g3_p90"], "darkred"),
            ),
        )

        # particle_types: expected PID classes present among target rows
        present = set()
        for i in range(self.nev_used):
            for f in TARGET_FIELDS:
                y = self.ev(f, i)
                if len(y) == 0:
                    continue
                present.update(y[y[:, PID] > 0, PID].astype(int).tolist())
        missing = sorted(EXPECTED_PIDS - present)
        gate = self.add_gate(
            "particle_types",
            "ground truth",
            "Particle types in the targets",
            "Representatives should cover the physics classes expected for the "
            f"sample: {sorted(EXPECTED_PIDS)} (e, mu, photon, K0L, charged hadron).",
            "missing: " + (", ".join(map(str, missing)) if missing else "none"),
            "PASS" if not missing else "WARN",
        )
        labels = sorted(EXPECTED_PIDS)
        self.bar_plot(
            gate,
            [str(p) for p in labels],
            [1 if p in present else 0 for p in labels],
            "present",
            colors=["#2ca02c" if p in present else "#ff7f0e" for p in labels],
        )

    def _evaluate_ratio_gate(self, ratios, median_range, frac_range, name):
        """Evaluate a ratio-based gate. Returns (status, observed string)."""
        lo, hi = median_range
        flo, fhi, fmin = frac_range
        n = len(ratios)
        if n == 0:
            return "WARN", "no measurements"
        med = float(np.median(ratios))
        frac = float(np.mean((ratios >= flo) & (ratios <= fhi)))
        if n < THRESHOLDS["min_fits"]:
            status = "WARN"
        else:
            status = "PASS" if (lo <= med <= hi and frac >= fmin) else "FAIL"
        return (
            status,
            f"n={n}, median({name})={med:.3f}, frac in [{flo}, {fhi}]={frac:.3f}",
        )

    def _closure_gate(
        self,
        gate_id,
        category,
        title,
        expectation,
        ratios,
        median_range,
        frac_range,
        name,
    ):
        """Generic ratio gate: median within range and fraction within [lo, hi]."""
        status, observed = self._evaluate_ratio_gate(ratios, median_range, frac_range, name)
        return self.add_gate(gate_id, category, title, expectation, observed, status)

    def _inclusive_hit_validation(self):
        """Tracker-hit pT (circle fit) and calo-hit energy closure vs truth.

        Collects per-particle closure measurements and writes the 4-panel
        diagnostic plot (hit_momentum_and_energy.png) used by gates tracker_hit_momentum/calorimeter_hit_energy.
        Returns {"track": ..., "calo": ...} with the raw truth/fitted arrays.
        """
        track_results = {"pT_truth": [], "pT_fit": []}
        calo_results = {"E_truth": [], "E_hit_sum": []}

        for iev in tqdm(range(self.nev_used), desc="inclusive hits"):
            pn_to_truth = {}
            for f in ["ytarget_track", "ytarget_cluster"]:
                y = self.ev(f, iev)
                for row in y:
                    pn = int(row[PN])
                    if pn > 0 and row[PID] > 0:
                        pn_to_truth[pn] = {"pt": row[PT], "energy": row[ENERGY]}

            # Tracker hits: circle fit per particle -> pT
            x = self.ev("X_hit_tracker", iev)
            y = self.ev("ytarget_hit_tracker", iev)
            if len(x) and len(y):
                mask = x[:, X_ELEMTYPE] != 0
                if np.any(mask):
                    pos = x[mask, X_HIT_POS_XY]
                    pns = y[mask, PN]
                    for pn in np.unique(pns):
                        if pn == 0 or pn not in pn_to_truth:
                            continue
                        hits = pos[pns == pn]
                        if len(hits) >= 5:
                            r = fit_circle(hits[:, 0], hits[:, 1])
                            if r and r < 1e5:
                                pT_fit = 0.0003 * self.bfield * r
                                track_results["pT_truth"].append(pn_to_truth[pn]["pt"])
                                track_results["pT_fit"].append(pT_fit)

                                if pT_fit / pn_to_truth[pn]["pt"] < 0.5 and len(track_results.get("bad_tracks", [])) < 5:
                                    if "bad_tracks" not in track_results:
                                        track_results["bad_tracks"] = []
                                    track_results["bad_tracks"].append(1)
                                    print(
                                        f"Track momentum differs from truth: event={iev} pn={pn} pT_truth={pn_to_truth[pn]['pt']:.3f} "
                                        f"pT_fit={pT_fit:.3f} R={r:.3f} mm"
                                    )

            # Calo hits: energy sum per particle
            x = self.ev("X_hit_calo", iev)
            y = self.ev("ytarget_hit_calo", iev)
            if len(x) and len(y):
                mask = x[:, X_ELEMTYPE] != 0
                if np.any(mask):
                    e_calo = x[mask, X_HIT_ENERGY]
                    pns = y[mask, PN]
                    for pn in np.unique(pns):
                        if pn == 0 or pn not in pn_to_truth:
                            continue
                        calo_results["E_truth"].append(pn_to_truth[pn]["energy"])
                        calo_results["E_hit_sum"].append(np.sum(e_calo[pns == pn]))

        for k in track_results:
            track_results[k] = np.array(track_results[k])
        for k in calo_results:
            calo_results[k] = np.array(calo_results[k])

        has_track = len(track_results["pT_truth"]) > 0
        has_calo = len(calo_results["E_truth"]) > 0

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 15))

        ax1.scatter(track_results["pT_truth"], track_results["pT_fit"], alpha=0.3, s=10)
        max_pt = max(float(np.max(track_results["pT_truth"])) if has_track else 0.0, 100)
        ax1.plot([0, max_pt], [0, max_pt], color="red", linestyle="--")
        ax1.set_xlabel("Truth particle pT [GeV]")
        ax1.set_ylabel("Fitted pT from Hits [GeV]")
        ax1.set_title("Momentum from tracker hits versus truth")
        ax1.set_xlim(0, max_pt)
        ax1.set_ylim(0, max_pt)

        ratio_pt = track_results["pT_fit"] / track_results["pT_truth"] if has_track else np.array([])
        mean_pt = float(np.mean(ratio_pt)) if has_track else float("nan")
        ax2.hist(
            ratio_pt,
            bins=100,
            range=(0, 2),
            alpha=0.7,
            color="darkorange",
            edgecolor="black",
        )
        ax2.axvline(1.0, color="red", linestyle="--")
        ax2.set_xlabel("pT_fit / pT_truth")
        ax2.set_title(f"Momentum Ratio (Mean: {mean_pt:.3f})")

        ax3.scatter(calo_results["E_truth"], calo_results["E_hit_sum"], alpha=0.3, s=10)
        max_e = max(float(np.max(calo_results["E_truth"])) if has_calo else 0.0, 100)
        ax3.plot([0, max_e], [0, max_e], color="red", linestyle="--")
        ax3.set_xlabel("Truth particle energy [GeV]")
        ax3.set_ylabel("Sum of Hit Energies [GeV]")
        ax3.set_title("Calorimeter hit energy versus truth")
        ax3.set_xlim(0, max_e)
        ax3.set_ylim(0, max_e)

        ratio_e = calo_results["E_hit_sum"] / calo_results["E_truth"] if has_calo else np.array([])
        mean_e = float(np.mean(ratio_e)) if has_calo else float("nan")
        ax4.hist(
            ratio_e,
            bins=100,
            range=(0, 2),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax4.axvline(1.0, color="red", linestyle="--")
        ax4.set_xlabel("Sum(E_hits) / E_gen")
        ax4.set_title(f"Energy Ratio (Mean: {mean_e:.3f})")

        print(f"Momentum Ratio Mean: {mean_pt:.3f}")
        print(f"Energy Ratio Mean: {mean_e:.3f}")

        # status text, consistent with the per-gate plots
        ratios_pt_clean = ratio_pt[np.isfinite(ratio_pt)] if len(ratio_pt) else ratio_pt
        ratios_e_clean = ratio_e[np.isfinite(ratio_e)] if len(ratio_e) else ratio_e
        h3_status, _ = self._evaluate_ratio_gate(
            ratios_pt_clean,
            THRESHOLDS["h3_median"],
            THRESHOLDS["h3_frac"],
            "pT_fit / pT_truth",
        )
        h4_status, _ = self._evaluate_ratio_gate(
            ratios_e_clean,
            THRESHOLDS["h4_median"],
            THRESHOLDS["h4_frac"],
            "E_hit_sum / E_truth",
        )
        combined = "PASS" if (h3_status == h4_status == "PASS") else ("FAIL" if "FAIL" in (h3_status, h4_status) else "WARN")
        colors = {"PASS": "#d4edda", "FAIL": "#f8d7da", "WARN": "#fff3cd"}
        text = (
            f"Particle momentum and energy from detector hits\n"
            "Expected: momentum from tracker hits and calorimeter hit energy agree with truth.\n"
            f"Observed: mean pT ratio={mean_pt:.3f}, mean energy ratio={mean_e:.3f}\n"
            f"Tracker momentum: {h3_status}; calorimeter energy: {h4_status}"
        )

        plt.suptitle(f"Particle momentum and energy from detector hits\nFile: {self.input_path.name}")
        fig.tight_layout(rect=[0, 0.14, 1, 1])
        fig.text(
            0.01,
            0.01,
            text,
            transform=fig.transFigure,
            fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor=colors.get(combined, "#ffffff"), alpha=0.85),
        )

        if self.save_plots:
            out_file = str(self.plots_dir / "hit_momentum_and_energy.png")
            fig.savefig(out_file, dpi=120, bbox_inches="tight")
            print(f"Saved gate plot to {out_file}")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        return {"track": track_results, "calo": calo_results}

    # --------------------------------------------------------------- reporting
    def summary(self):
        n_pass = sum(g.status == "PASS" for g in self.gates)
        n_fail = sum(g.status == "FAIL" for g in self.gates)
        n_warn = sum(g.status == "WARN" for g in self.gates)
        n_skip = sum(g.status == "SKIP" for g in self.gates)
        overall = "FAIL" if n_fail else "PASS"
        return overall, n_pass, n_fail, n_warn, n_skip

    def write_report(self, json_path):
        overall, n_pass, n_fail, n_warn, n_skip = self.summary()
        plots = sorted(g.plot for g in self.gates if g.plot)
        if (self.plots_dir / "hit_momentum_and_energy.png").exists():
            plots.append("hit_momentum_and_energy.png")
        report = {
            "file": str(self.input_path),
            "detector": self.detector,
            "bfield": self.bfield,
            "events_used": self.nev_used,
            "overall": overall,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_warn": n_warn,
            "n_skip": n_skip,
            "plots": plots,
            "gates": [asdict(g) for g in self.gates],
            "alignment_metrics": self.alignment_metrics,
        }
        if json_path is not None:
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)

        print(f"\n=== Validation report: {self.input_path} ===")
        print(f"detector={self.detector} bfield={self.bfield} events_used={self.nev_used}")
        print(f"{'check':<30} {'status':<6} result")
        print("-" * 90)
        for g in self.gates:
            print(f"{g.gate_id:<30} {g.status:<6} {g.observed}")
        print("-" * 90)
        print(f"PASS={n_pass} FAIL={n_fail} WARN={n_warn} SKIP={n_skip} overall={overall}")
        if self.save_plots:
            print(f"plots: {self.plots_dir}")
        if json_path is not None:
            print(f"report: {json_path}")
        return n_fail == 0

    def run(self):
        if self.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.plots_dir / "validation_report.json" if self.save_plots else None
        if not self.load():
            return self.write_report(report_path)
        self.gate_schema()
        if any(g.status == "FAIL" for g in self.gates):
            return self.write_report(report_path)
        if self.nev > 0:
            self.gate_hits()
            self.gate_visualization()
            self.gate_hit_rep_consistency()
            self.gate_tracks_clusters()
            self.gate_element_target_alignment()
            self.gate_targets()
            self.gate_target_definition()
            self.gate_baseline_pf()
            self.gate_truth()
        return self.write_report(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet file or directory of parquet files to validate",
    )
    parser.add_argument(
        "--detector",
        choices=list(EDM4HEP.DETECTORS.keys()),
        required=True,
        help=f"Detector scenario from mlpf/conf.py ({', '.join(EDM4HEP.DETECTORS.keys())})",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to check per file (default: all)",
    )
    parser.add_argument("--plots-dir", default="validation_plots", help="Directory for plots and validation_report.json")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            print(f"No parquet files found in {input_path}")
            sys.exit(1)
    else:
        files = [input_path]

    all_ok = True
    for f in files:
        detector = args.detector
        plots_dir = Path(args.plots_dir) / f.stem if len(files) > 1 else Path(args.plots_dir)
        validator = ParquetValidator(f, detector, args.max_events, plots_dir)
        ok = validator.run()
        all_ok &= ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
