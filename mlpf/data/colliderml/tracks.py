# Build MLPF-ready track features from ColliderML ACTS perigee parameters.
#
# track_features_cml returns a per-track feature dict with keys: type (=1), pt [GeV]
# (= |1/qop| * sin(theta), qop verified in GeV^-1), eta (= -ln tan(theta/2)), sin_phi, cos_phi,
# p [GeV] (= |1/qop|, the energy-like scale for the log-ratio target), d0, z0, theta, qop, and
# n_meas (= len(hit_ids) per track).
#
# The converter (postprocessing.py) packs these keys into the 17-column clustered-view union
# layout; see that file and X_FEATURES[Dataset.COLLIDERML] in mlpf/conf.py for the column map.
import math
from typing import Any, Dict

import awkward as ak
import numpy as np


def _sin_eta_from_theta(theta: np.ndarray):
    # theta in (0, pi) as ACTS perigee parameter; both sin and eta derived directly.
    # Guard against degenerate 0 or pi track fits.
    eps = 1e-9
    theta_c = np.clip(theta, eps, math.pi - eps)
    eta = -np.log(np.tan(theta_c / 2))
    return np.sin(theta_c), eta


def track_features_cml(tracks_ev: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert one event's ACTS perigee-parameter record to the MLPF track feature dict.

    `tracks_ev` is a per-event dict of awkward arrays with fields d0, z0, phi, theta, qop,
    majority_particle_id, hit_ids, track_id.

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
        "n_meas": n_meas.astype(np.float32),
    }
