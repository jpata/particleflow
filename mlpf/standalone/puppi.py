import numpy as np
import math
from scipy.special import erf


def deltaR_matrix(eta, cosphi, sinphi):
    """
    Compute N xN Delta R_ij matrix given N eta, cosphi, sinphi
    """
    phi = np.arctan2(sinphi, cosphi)
    d_eta = eta[:, None] - eta[None, :]
    d_phi = phi[:, None] - phi[None, :]
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(d_eta**2 + d_phi**2)


def compute_alpha_central(pt, eta, cosphi, sinphi, is_charged_pv, R0=0.4):
    """
    alpha computed ONLY using charged prompt particles.
    """
    dr = deltaR_matrix(eta, cosphi, sinphi)

    # mask for neighbors: charged PV and within 0 < R < R0 only
    pu_mask = is_charged_pv[None, :]
    mask = (dr > 0) & (dr < R0) & pu_mask

    activity = np.sum((np.nan_to_num(pt[None, :] / dr) ** 2) * mask, axis=1)

    alpha = np.nan_to_num(np.log(activity), neginf=0)

    return alpha


def compute_puppi_weights(pt, eta, cosphi, sinphi, is_charged, is_from_PV, R0=0.4, med_rms_min_pt=0.1):
    """
    Central-region PUPPI (tracker available and no extrapolation done)
    Inputs:
    pt, eta, cosphi, sinphi, is_charged, is_from_PV are 1D arrays for particles
    is_from_PV is only meaningful for charged particles
    In CMSSW, is_from_PV is determined by CHS (i.e. charged particles with small dz and dxy are assumed to be from PV)
    R0 is the radius used in the computation of alpha
    med_rms_min_pt is the minimum pT of charged particles used in the median and rms alpha computation, set to 0.1, same as CMSSW
    """

    # Identify categories
    is_charged_pu = is_charged & (~is_from_PV)
    is_charged_pv = is_charged & is_from_PV
    pass_med_rms_min_pt = pt > med_rms_min_pt

    # Step 1: compute alpha
    alpha = compute_alpha_central(pt, eta, cosphi, sinphi, is_charged_pv, R0)

    # Step 2: reference distribution from charged PU
    # Notice we apply a med_rms_min_pt cut on the pt and remove alpha==0 elements
    alpha_ref = alpha[is_charged_pu & pass_med_rms_min_pt & (alpha != 0)]

    median = np.median(alpha_ref)
    rms = np.sqrt(np.mean(np.square(alpha_ref)))

    # Step 3: s is the sqrt of χ²
    s = (alpha - median) / rms

    # Step 4: compute weight
    weights = np.ones_like(pt)
    mask = s < 0
    s[mask] = 0
    weights = erf(s / math.sqrt(2))

    # Step 5: set weights to be 1 or 0 for charged particles
    weights[is_charged_pv] = 1.0
    weights[is_charged_pu] = 0.0

    return weights, s, alpha
