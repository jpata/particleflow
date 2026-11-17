"""Calorimeter-frame (eta, phi) coordinates for HEPTv2 bucketing.

HEPTv2 hashes elements into attention buckets by the (eta, phi) in X columns 2-4, and its learned
positional encoding reads the same coordinates. By default a track sits at its direction at the
vertex. A charged track bends in the solenoid field before it reaches the calorimeter (a 1 GeV track
in a 3 T field turns by ~0.6 rad in phi before r = 1.3 m), so it can be far from its own clusters.

heptv2.track_hash_at_calo (TrackCaloEntry) moves each track to the (eta, phi) of its calorimeter
entry point, seen from the origin: helix-extrapolated from the track parameters (d0, z0, tanLambda,
omega) to the middle of the ECAL on the key4hep-style datasets, read from the eta_ecal/phi_ecal
features on CMS.

heptv2.cluster_hash_at_centroid (ClusterCentroid) moves each cluster to the (eta, phi) of its
centroid position. On the key4hep PF datasets (CLIC, CLD, IDEA, MAIA) the cluster eta/phi features
are Pandora's iTheta/phi, the direction of the shower axis, which is not where the cluster sits (on
CLIC ttbar they differ from the centroid by a median of 0.20 in eta and 0.35 rad in phi); CMS and
ColliderML cluster features are already positions.

Fraction of cluster energy with a track within dR < 0.05 (200 ttbar events per detector; CMS: ECAL
clusters, 50 events with pileup):

    detector    tracks at vertex     tracks at calo entry    tracks at calo entry,
                                                             clusters at centroid
    CLIC        0.11                 0.09                    0.50
    CLD         0.12                 0.10                    0.57
    CMS         0.50                 0.55                    (already positions)

Both only change the X_features the backbone sees; the input encoder and the output heads keep the
original features.
"""

import torch

from mlpf.conf import X_FEATURES, Dataset, SOURCE_IDS

TRACK_ELEMTYPE = 1

# ECAL ((barrel inner, outer radius), (endcap min, max |z|)) [mm] from the detector compact files: ColliderML =
# OpenDataDetector (xml/OpenDataDetectorEnvelopes.xml: ecal_b_rmin/rmax, ecal_e_min_z/max_z), CLIC = k4geo CLIC_o3_v14,
# CLD = k4geo CLD_o2_v07 (ECalBarrel_inner/outer_radius, ECalEndcap_min/max_z).
ECAL_GEOMETRY_MM = {
    Dataset.COLLIDERML.value: ((1250.0, 1500.0), (3200.0, 3450.0)),
    Dataset.CLIC.value: ((1500.0, 1702.0), (2307.0, 2509.0)),
    Dataset.CLD.value: ((2150.0, 2352.0), (2307.0, 2509.0)),
}
# (barrel radius, endcap |z|) [mm] tracks are extrapolated to: the middle of the ECAL, where the cluster centroids sit
# (shower depth). The front face puts tracks further from their clusters: on CLIC ttbar 8% less cluster energy has a
# track within dR < 0.05 (CLD 4%); mid-depth is within 1% of the best radius on all three detectors.
CALO_ENTRY_GEOMETRY_MM = {dataset: (sum(radii) / 2, sum(z) / 2) for dataset, (radii, z) in ECAL_GEOMETRY_MM.items()}
# IDEA (k4geo IDEA_o1_v03, DectDimensions_IDEA_o1_v03.xml) has no separate ECAL: its fibre dual-readout calorimeter
# towers start at r = |z| = 2500 mm (FiberDRCalo_barrel_z_start, FiberDRCalo_endcap_z_start) and are 2 m deep without
# depth segmentation. Its cluster centroids sit at that front face (ttbar median r 2501 mm, endcap |z| 2502 mm), so
# tracks go to the front face.
CALO_ENTRY_GEOMETRY_MM[Dataset.IDEA.value] = (2500.0, 2500.0)

# PF datasets whose tracks carry the helix parameters, by name in X_FEATURES.
HELIX_DATASETS = (Dataset.COLLIDERML.value, Dataset.CLIC.value, Dataset.CLD.value, Dataset.IDEA.value, Dataset.MAIA.value)
HELIX_COLUMNS = ("z0", "tanlambda", "omega", "d0")
# Datasets whose tracks carry their calorimeter entry (eta, phi) as features.
PRECOMPUTED_COLUMNS = {Dataset.CMS.value: ("eta_ecal", "phi_ecal")}

CLUSTER_ELEMTYPE = 2
# key4hep PF datasets: cluster eta/phi are Pandora's shower-axis direction, the centroid is in position.x/y/z.
CENTROID_DATASETS = (Dataset.CLIC.value, Dataset.CLD.value, Dataset.IDEA.value, Dataset.MAIA.value)
CENTROID_COLUMNS = ("position.x", "position.y", "position.z")


def calo_entry_eta_phi(phi0, z0, tan_lambda, omega, radius, endcap_z, d0=None):
    """(eta, phi, reached) of the point where a track enters the calorimeter, seen from the origin.

    The track is a helix with its point of closest approach to the beam axis (perigee) at
    (-d0 sin phi0, d0 cos phi0, z0), azimuth phi0 there, signed transverse curvature omega [1/mm] and
    tan_lambda = pz / pt (d0 = None: d0 = 0). This is the EDM4hep/LCIO track-state convention with the
    reference point at the origin, which the CLIC, CLD, MAIA and ColliderML (ACTS) tracks follow: their
    tracker hits, or their AtFirstHit state, lie on this helix and not on the one with -d0. The direction
    turns by -omega per unit transverse arc, the sign that puts the most cluster energy around the
    tracks on ColliderML, CLIC and CLD; the opposite sign puts less there than the vertex direction does.

    The helix ends at the first of the barrel cylinder r = radius [mm] and the endcap planes
    |z| = endcap_z [mm], following the track forward from its perigee. A track that curls up inside the
    barrel radius can only reach an endcap plane (at any transverse radius: the endcap's inner and outer
    radii are not checked). reached is False for a track that reaches neither surface, or whose perigee
    lies at or beyond an endcap plane (|z0| >= endcap_z) or outside the barrel (|d0| >= radius); its eta
    and phi are then its direction at the perigee (asinh(tan_lambda), phi0).

    After a transverse arc s the track lies at the perigee plus a chord of signed length
    q = 2 sin(omega * s / 2) / omega along azimuth phi0 - omega * s / 2, at a transverse distance
    r^2 = d0^2 + (1 - omega * d0) q^2 from the beam axis, which gives the barrel arc in closed form.
    """
    eta, sin_phi, cos_phi, reached = _calo_entry_eta_direction(
        torch.sin(phi0.float()), torch.cos(phi0.float()), z0, tan_lambda, omega, radius, endcap_z, d0
    )
    return eta, torch.atan2(sin_phi, cos_phi), reached


def _calo_entry_eta_direction(sin_phi0, cos_phi0, z0, tan_lambda, omega, radius, endcap_z, d0=None):
    """Extrapolate a unit transverse direction without inverse-trig round trips."""
    sin_phi0, cos_phi0, z0, tan_lambda, omega = (t.float() for t in (sin_phi0, cos_phi0, z0, tan_lambda, omega))
    d0 = torch.zeros_like(z0) if d0 is None else d0.float()
    curvature = omega.abs().clamp_min(1e-12)
    no_hit = torch.full_like(curvature, float("inf"))

    # barrel: q^2 = (radius^2 - d0^2) / (1 - omega d0), and sin(|omega| s / 2) = |omega| |q| / 2
    stretch = 1 - omega * d0
    inside = (d0.abs() < radius) & (stretch > 0)
    chord_sq = (radius**2 - d0**2) / torch.where(inside, stretch, torch.ones_like(stretch))
    half_chord = curvature * chord_sq.clamp_min(0).sqrt() / 2
    arc_barrel = torch.where(inside & (half_chord < 1), 2 * _asin(half_chord.clamp(max=1.0)) / curvature, no_hit)

    forward = tan_lambda.abs() > 1e-6
    safe_tan_lambda = torch.where(forward, tan_lambda, torch.ones_like(tan_lambda))
    arc_endcap = (torch.sign(tan_lambda) * endcap_z - z0) / safe_tan_lambda
    arc_endcap = torch.where(forward & (arc_endcap > 0), arc_endcap, no_hit)

    arc = torch.minimum(arc_barrel, arc_endcap)
    reached = torch.isfinite(arc) & (z0.abs() < endcap_z) & (d0.abs() < radius)
    arc = torch.where(reached, arc, torch.zeros_like(arc))

    half_turn = omega * arc / 2
    safe_omega = torch.where(omega == 0, torch.ones_like(omega), omega)
    # signed: past a full turn (endcap entries only) the chord points backwards
    chord = torch.where(half_turn.abs() > 1e-6, 2 * torch.sin(half_turn) / safe_omega, arc)
    sin_turn, cos_turn = torch.sin(half_turn), torch.cos(half_turn)
    x = -d0 * sin_phi0 + chord * (cos_phi0 * cos_turn + sin_phi0 * sin_turn)
    y = d0 * cos_phi0 + chord * (sin_phi0 * cos_turn - cos_phi0 * sin_turn)
    z = z0 + arc * tan_lambda
    eta = _asinh(z / torch.sqrt(x.square() + y.square()).clamp_min(1.0))
    sin_phi, cos_phi = _unit_direction(x, y)
    eta = torch.where(reached, eta, _asinh(tan_lambda))
    sin_phi = torch.where(reached, sin_phi, sin_phi0)
    cos_phi = torch.where(reached, cos_phi, cos_phi0)
    return eta, sin_phi, cos_phi, reached


def _asin(value):
    """FP32 asin on [-1, 1] using only portable elementwise ONNX operations.

    Reduce |x| > 1/2 with asin(x) = pi/2 - 2 asin(sqrt((1-x)/2)).
    On [0, 1/2], evaluate the degree-21 Taylor polynomial in Horner form;
    coefficients are binomial(2n,n) / (4**n * (2n+1)). Its truncation error
    is below 2.4e-9 radians, including the factor two after range reduction.
    Eager and export use the same arithmetic to avoid different hash inputs.
    """
    magnitude = value.abs()
    reduced = torch.where(magnitude > 0.5, torch.sqrt((1 - magnitude) * 0.5), magnitude)
    squared = reduced.square()
    coefficients = (
        1.0,
        1 / 6,
        3 / 40,
        5 / 112,
        35 / 1152,
        63 / 2816,
        231 / 13312,
        143 / 10240,
        6435 / 557056,
        12155 / 1245184,
        46189 / 5505024,
    )
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[1:-1]):
        polynomial = coefficient + squared * polynomial
    # Add the leading x separately to retain precision near zero.
    angle = reduced + reduced * squared * polynomial
    # Compensate for rounding pi/2 to FP32, especially just above x=1/2.
    upper = (1.5707963705062866 - 2 * angle) - 4.3711390001862426e-8
    angle = torch.where(magnitude > 0.5, upper, angle)
    return torch.where(value < 0, -angle, angle)


def _unit_direction(x, y):
    """Return (sin(phi), cos(phi)); the zero vector has atan2(0,0)'s direction.

    Scale first to avoid overflow/underflow for non-unit input directions.
    """
    scale = torch.maximum(x.abs(), y.abs())
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x, y = x / scale, y / scale
    norm = torch.sqrt(x.square() + y.square())
    nonzero = norm > 0
    norm = torch.where(nonzero, norm, torch.ones_like(norm))
    return y / norm, torch.where(nonzero, x / norm, torch.ones_like(x))


def _asinh(value):
    """Numerically stable asinh using operators supported by the legacy ONNX exporter."""
    magnitude = value.abs()
    squared = magnitude.square()
    return value.sign() * torch.log1p(magnitude + squared / (torch.sqrt(1 + squared) + 1))


def _track_column_names(dataset):
    # X_FEATURES names shared columns "<track name> | <cluster name>".
    return [name.split("|")[0].strip().lower() for name in X_FEATURES[dataset]]


def _cluster_column_names(dataset):
    return [name.split("|")[-1].strip().lower() for name in X_FEATURES[dataset]]


def _column_indices(names, wanted, dataset, flag):
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"heptv2.{flag} needs the X_FEATURES columns {missing}, which dataset {dataset} does not have")
    return [names.index(name) for name in wanted]


def _replace_coordinates(X_features, replace, eta, sin_phi, cos_phi):
    out = X_features.clone()
    out[..., 2] = torch.where(replace, eta.to(X_features.dtype), X_features[..., 2])
    out[..., 3] = torch.where(replace, sin_phi.to(X_features.dtype), X_features[..., 3])
    out[..., 4] = torch.where(replace, cos_phi.to(X_features.dtype), X_features[..., 4])
    return out


class TrackCaloEntry:
    """Replaces the track rows' (eta, sin_phi, cos_phi) X columns 2-4 by their calorimeter entry point.

    Stateless (no parameters or buffers), so enabling it leaves the model's state dict unchanged.
    """

    def __init__(self, dataset, radius_mm=None, endcap_z_mm=None):
        self.dataset = dataset
        names = _track_column_names(dataset)
        if dataset in PRECOMPUTED_COLUMNS:
            self.mode = "precomputed"
            self.columns = _column_indices(names, PRECOMPUTED_COLUMNS[dataset], dataset, "track_hash_at_calo")
        elif dataset in HELIX_DATASETS:
            self.mode = "helix"
            self.columns = _column_indices(names, HELIX_COLUMNS, dataset, "track_hash_at_calo")
            default = CALO_ENTRY_GEOMETRY_MM.get(dataset, (None, None))
            self.radius = radius_mm if radius_mm is not None else default[0]
            self.endcap_z = endcap_z_mm if endcap_z_mm is not None else default[1]
            if self.radius is None or self.endcap_z is None:
                raise ValueError(
                    f"heptv2.track_hash_at_calo has no calorimeter geometry for dataset {dataset}: "
                    "set heptv2.track_hash_calo_radius_mm and heptv2.track_hash_calo_endcap_z_mm"
                )
        else:
            supported = sorted(HELIX_DATASETS + tuple(PRECOMPUTED_COLUMNS))
            raise ValueError(f"heptv2.track_hash_at_calo is not supported for dataset {dataset} (supported: {supported})")

    def __repr__(self):
        if self.mode == "helix":
            return f"TrackCaloEntry({self.dataset}, helix, radius={self.radius} mm, endcap_z={self.endcap_z} mm)"
        return f"TrackCaloEntry({self.dataset}, precomputed columns {self.columns})"

    def eta_phi(self, X_features):
        """(eta, phi, defined) of every row's calorimeter entry, computed as if the row were a track."""
        if self.mode == "helix":
            z0, tan_lambda, omega, d0 = (X_features[..., column] for column in self.columns)
            phi0 = torch.atan2(X_features[..., 3], X_features[..., 4]).float()
            # tracks that reach no calorimeter surface keep their vertex direction
            return calo_entry_eta_phi(phi0, z0, tan_lambda, omega, self.radius, self.endcap_z, d0=d0)
        eta, phi = (X_features[..., column].float() for column in self.columns)
        # tracks without an ECAL extrapolation carry eta_ecal = phi_ecal = 0: they keep their vertex direction (as would
        # non-finite values)
        return eta, phi, torch.isfinite(eta) & torch.isfinite(phi) & ((eta != 0) | (phi != 0))

    @torch.no_grad()
    def __call__(self, X_features, source_id=None):
        if self.mode == "helix":
            z0, tan_lambda, omega, d0 = (X_features[..., column] for column in self.columns)
            sin_phi0, cos_phi0 = _unit_direction(X_features[..., 4].float(), X_features[..., 3].float())
            eta, sin_phi, cos_phi, defined = _calo_entry_eta_direction(sin_phi0, cos_phi0, z0, tan_lambda, omega, self.radius, self.endcap_z, d0)
        else:
            eta, phi, defined = self.eta_phi(X_features)
            sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        return _replace_coordinates(X_features, (X_features[..., 0] == TRACK_ELEMTYPE) & defined, eta, sin_phi, cos_phi)


class MultiSourceTrackCaloEntry:
    """Use each event's detector layout and calorimeter geometry for track coordinates."""

    def __init__(self, sources, radius_mm=None, endcap_z_mm=None):
        self.entries = {SOURCE_IDS[source]: TrackCaloEntry(source, radius_mm=radius_mm, endcap_z_mm=endcap_z_mm) for source in sources}

    @torch.no_grad()
    def __call__(self, X_features, source_id):
        if source_id is None:
            raise ValueError("Multi-source calorimeter hashing needs each event's source_id")
        if source_id.numel() != X_features.shape[0]:
            raise ValueError("Calorimeter hashing needs one source_id per event")
        source_id = source_id.to(device=X_features.device).view(-1, 1, 1)
        out = X_features
        for source, entry in self.entries.items():
            out = torch.where(source_id == source, entry(X_features), out)
        return out


class ClusterCentroid:
    """Replaces the cluster rows' (eta, sin_phi, cos_phi) X columns 2-4 by the direction of their centroid.

    Stateless (no parameters or buffers), so enabling it leaves the model's state dict unchanged.
    """

    def __init__(self, dataset):
        if dataset not in CENTROID_DATASETS:
            raise ValueError(f"heptv2.cluster_hash_at_centroid is not supported for dataset {dataset} (supported: {sorted(CENTROID_DATASETS)})")
        self.dataset = dataset
        names = _cluster_column_names(dataset)
        self.columns = _column_indices(names, CENTROID_COLUMNS, dataset, "cluster_hash_at_centroid")

    def __repr__(self):
        return f"ClusterCentroid({self.dataset}, position columns {self.columns})"

    @torch.no_grad()
    def __call__(self, X_features):
        x, y, z = (X_features[..., column].float() for column in self.columns)
        eta = _asinh(z / torch.sqrt(x.square() + y.square()).clamp_min(1.0))
        sin_phi, cos_phi = _unit_direction(x, y)
        return _replace_coordinates(X_features, X_features[..., 0] == CLUSTER_ELEMTYPE, eta, sin_phi, cos_phi)
