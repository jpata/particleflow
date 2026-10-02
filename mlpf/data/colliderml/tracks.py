# Build MLPF-ready track features from ColliderML ACTS perigee parameters.
#
# track_features_cml returns a per-track feature dict with keys: type (=1), pt [GeV]
# (= |1/qop| * sin(theta), qop verified in GeV^-1), eta (= -ln tan(theta/2)), sin_phi, cos_phi,
# p [GeV] (= |1/qop|, the energy-like scale for the log-ratio target), d0, z0, theta, qop, and
# n_meas (= len(hit_ids) per track). It also derives the EDM4hep track-state parameters the
# key4hep converter reads directly: tanLambda (= 1/tan(theta) = sinh(eta), so that
# eta = arcsinh(tanLambda) holds like in key4hep), omega (= q/pT in 1/mm, computed from qop
# with the ODD solenoid field; this is key4hep's track_pt inverted), and
# radiusOfInnermostHit (the transverse radius of the track's innermost tracker hit, the
# AtFirstHit reference-point radius in key4hep).
#
# The converter (postprocessing.py) packs these keys into the 17-column clustered-view union
# layout; see that file and X_FEATURES[Dataset.COLLIDERML] in mlpf/conf.py for the column map.
import math
from typing import Any, Dict

import awkward as ak
import numpy as np

from mlpf.conf import EDM4HEP

# ODD solenoid field [T] from the detector registry (colliderml entry); used to express the
# signed curvature omega = q/pT in 1/mm with the key4hep convention (their track_pt).
B_FIELD = EDM4HEP.DETECTORS["colliderml"].b_field


def _sin_eta_from_theta(theta: np.ndarray):
    # theta in (0, pi) as ACTS perigee parameter; both sin and eta derived directly.
    # Guard against degenerate 0 or pi track fits.
    eps = 1e-9
    theta_c = np.clip(theta, eps, math.pi - eps)
    eta = -np.log(np.tan(theta_c / 2))
    return np.sin(theta_c), eta


def track_features_cml(tracks_ev: Dict[str, Any], tracker_ev: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert one event's ACTS perigee-parameter record to the MLPF track feature dict.

    `tracks_ev` is a per-event dict of awkward arrays with fields d0, z0, phi, theta, qop,
    majority_particle_id, hit_ids, track_id. `tracker_ev` is the event's tracker-hit record;
    only its x/y positions are read (joined through tracks.hit_ids) to compute
    radiusOfInnermostHit.

    Returns a dict of numpy arrays of length n_tracks with the keys listed above plus
    `pt` and `p` and `eta`.
    """
    d0 = ak.to_numpy(tracks_ev["d0"])
    z0 = ak.to_numpy(tracks_ev["z0"])
    phi = ak.to_numpy(tracks_ev["phi"])
    theta = ak.to_numpy(tracks_ev["theta"])
    qop = ak.to_numpy(tracks_ev["qop"])
    n_meas = ak.to_numpy(ak.num(tracks_ev["hit_ids"]))

    # protect against degenerate tracks: qop==0 would give infinite p; clamp to the largest
    # physically meaningful scale measured in this release (~10^4 GeV) rather than inf/nan so
    # the input pipeline stays finite
    safe_qop = np.where(qop == 0.0, np.sign(qop) * 1e-3, qop)
    safe_qop = np.where(safe_qop == 0.0, 1e-3, safe_qop)  # qop exactly 0
    p = np.abs(1.0 / safe_qop)
    p = np.minimum(p, 1.0e4)
    sin_theta, eta = _sin_eta_from_theta(theta)
    pt = p * sin_theta
    # EDM4hep track-state parameters, derived from the ACTS perigee set:
    # tanLambda = pz/pt = 1/tan(theta) = sinh(eta)  (key4hep: eta = arcsinh(tanLambda))
    # omega = q/pT [1/mm] via pt [GeV] = 3e-4 * B[T] / omega (key4hep track_pt inverted)
    tan_lambda = np.sinh(eta)
    omega = np.sign(safe_qop) * 3.0e-4 * B_FIELD / pt

    # radiusOfInnermostHit: transverse radius of the track's innermost tracker hit (key4hep
    # reads the AtFirstHit track-state reference point). hit_ids are row indices into the
    # event's tracker_hits table, so this is a flat gather + segmented min; a hitless track
    # gets 0.
    n_track = len(d0)
    hit_x = ak.to_numpy(tracker_ev["x"])
    hit_y = ak.to_numpy(tracker_ev["y"])
    flat_hids = np.asarray(ak.to_numpy(ak.flatten(tracks_ev["hit_ids"], axis=None)), dtype=np.int64)
    r_first = np.zeros(n_track, dtype=np.float64)
    if len(flat_hids):
        in_range = (flat_hids >= 0) & (flat_hids < len(hit_x))  # defensive; release data is in range
        seg = np.repeat(np.arange(n_track, dtype=np.int64), n_meas.astype(np.int64))[in_range]
        r = np.hypot(hit_x[flat_hids[in_range]], hit_y[flat_hids[in_range]])
        r_seg = np.full(n_track, np.inf, dtype=np.float64)
        np.minimum.at(r_seg, seg, r)
        r_first = np.where(np.isfinite(r_seg), r_seg, 0.0)

    return {
        "type": np.ones(len(d0), dtype=np.float32),
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "sin_phi": np.sin(phi).astype(np.float32),
        "cos_phi": np.cos(phi).astype(np.float32),
        "p": p.astype(np.float32),
        "d0": d0.astype(np.float32),
        "z0": z0.astype(np.float32),
        "theta": theta.astype(np.float32),
        "qop": qop.astype(np.float32),
        "tanLambda": tan_lambda.astype(np.float32),
        "omega": omega.astype(np.float32),
        "radiusOfInnermostHit": r_first.astype(np.float32),
        "n_meas": n_meas.astype(np.float32),
    }
