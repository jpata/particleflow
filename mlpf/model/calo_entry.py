"""Calorimeter-frame (eta, phi) coordinates for HEPTv2 bucketing.

HEPTv2 hashes elements into attention buckets by the (eta, phi) in X columns 2-4, and its learned
positional encoding reads the same coordinates. By default a track sits at its direction at the
vertex. A charged track bends in the solenoid field before it reaches the calorimeter (a 1 GeV track
in a 3 T field turns by ~0.6 rad in phi before r = 1.3 m), so it can be far from its own clusters.

heptv2.track_hash_at_calo (TrackCaloEntry) moves each track to the (eta, phi) of its calorimeter
entry point, seen from the origin: helix-extrapolated from the track parameters (z0, tanLambda,
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
HELIX_COLUMNS = ("z0", "tanlambda", "omega")
# Datasets whose tracks carry their calorimeter entry (eta, phi) as features.
PRECOMPUTED_COLUMNS = {Dataset.CMS.value: ("eta_ecal", "phi_ecal")}

CLUSTER_ELEMTYPE = 2
# key4hep PF datasets: cluster eta/phi are Pandora's shower-axis direction, the centroid is in position.x/y/z.
CENTROID_DATASETS = (Dataset.CLIC.value, Dataset.CLD.value, Dataset.IDEA.value, Dataset.MAIA.value)
CENTROID_COLUMNS = ("position.x", "position.y", "position.z")


def calo_entry_eta_phi(phi0, z0, tan_lambda, omega, radius, endcap_z):
    """(eta, phi) of the point where a track enters the calorimeter, seen from the origin.

    The track is a helix from (0, 0, z0) (d0 neglected) with initial azimuth phi0, signed
    transverse curvature omega [1/mm] and tan_lambda = pz / pt. It ends at the first of the
    barrel cylinder r = radius [mm] and the endcap planes |z| = endcap_z [mm]; a track that curls
    up inside the barrel radius can only reach an endcap, and one that reaches neither keeps its
    vertex direction. After a transverse arc s the entry point lies at signed chord length
    2 sin(omega * s / 2) / omega along azimuth phi0 - omega * s / 2. This is the sign of the
    EDM4hep/LCIO omega convention (and of ColliderML's q/pT-based omega): it puts the most
    cluster energy around the tracks on ColliderML, CLIC and CLD; the opposite sign puts less
    there than the vertex direction does.
    """
    phi0, z0, tan_lambda, omega = (t.float() for t in (phi0, z0, tan_lambda, omega))
    curvature = omega.abs().clamp_min(1e-12)
    no_hit = torch.full_like(curvature, float("inf"))

    half_chord = curvature * radius / 2
    arc_barrel = torch.where(half_chord < 1, 2 * torch.asin(half_chord.clamp(max=1.0)) / curvature, no_hit)

    forward = tan_lambda.abs() > 1e-6
    safe_tan_lambda = torch.where(forward, tan_lambda, torch.ones_like(tan_lambda))
    arc_endcap = (torch.sign(tan_lambda) * endcap_z - z0) / safe_tan_lambda
    arc_endcap = torch.where(forward & (arc_endcap > 0), arc_endcap, no_hit)

    arc = torch.minimum(arc_barrel, arc_endcap)
    arc = torch.where(torch.isfinite(arc), arc, torch.zeros_like(arc))

    half_turn = omega * arc / 2
    safe_omega = torch.where(omega == 0, torch.ones_like(omega), omega)
    chord = torch.where(half_turn.abs() > 1e-6, 2 * torch.sin(half_turn) / safe_omega, arc)
    # past half a turn the chord is negative: the entry point is on the opposite side
    direction = phi0 - half_turn + torch.where(chord < 0, torch.full_like(chord, torch.pi), torch.zeros_like(chord))
    phi = torch.atan2(torch.sin(direction), torch.cos(direction))
    z = z0 + arc * tan_lambda
    eta = torch.asinh(z / chord.abs().clamp_min(1.0))
    return eta, phi


def _track_column_names(dataset):
    # X_FEATURES names shared columns "<track name> | <cluster name>".
    return [name.split("|")[0].strip().lower() for name in X_FEATURES[dataset]]


def _cluster_column_names(dataset):
    return [name.split("|")[-1].strip().lower() for name in X_FEATURES[dataset]]


def _replace_coordinates(X_features, replace, eta, phi):
    out = X_features.clone()
    out[..., 2] = torch.where(replace, eta.to(X_features.dtype), X_features[..., 2])
    out[..., 3] = torch.where(replace, torch.sin(phi).to(X_features.dtype), X_features[..., 3])
    out[..., 4] = torch.where(replace, torch.cos(phi).to(X_features.dtype), X_features[..., 4])
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
            self.columns = [names.index(name) for name in PRECOMPUTED_COLUMNS[dataset]]
        elif dataset in HELIX_DATASETS:
            self.mode = "helix"
            self.columns = [names.index(name) for name in HELIX_COLUMNS]
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
            z0, tan_lambda, omega = (X_features[..., column] for column in self.columns)
            phi0 = torch.atan2(X_features[..., 3], X_features[..., 4]).float()
            eta, phi = calo_entry_eta_phi(phi0, z0, tan_lambda, omega, self.radius, self.endcap_z)
            return eta, phi, torch.ones_like(eta, dtype=torch.bool)
        eta, phi = (X_features[..., column].float() for column in self.columns)
        # tracks without an ECAL extrapolation carry eta_ecal = phi_ecal = 0: they keep their vertex direction
        return eta, phi, (eta != 0) | (phi != 0)

    @torch.no_grad()
    def __call__(self, X_features, source_id=None):
        eta, phi, defined = self.eta_phi(X_features)
        return _replace_coordinates(X_features, (X_features[..., 0] == TRACK_ELEMTYPE) & defined, eta, phi)


class MultiSourceTrackCaloEntry:
    """Use each event's detector layout and calorimeter geometry for track coordinates."""

    def __init__(self, sources, radius_mm=None, endcap_z_mm=None):
        self.entries = {
            SOURCE_IDS[source]: TrackCaloEntry(source, radius_mm=radius_mm, endcap_z_mm=endcap_z_mm)
            for source in sources
        }

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
        self.columns = [names.index(name) for name in CENTROID_COLUMNS]

    def __repr__(self):
        return f"ClusterCentroid({self.dataset}, position columns {self.columns})"

    @torch.no_grad()
    def __call__(self, X_features):
        x, y, z = (X_features[..., column].float() for column in self.columns)
        eta = torch.asinh(z / torch.hypot(x, y).clamp_min(1.0))
        phi = torch.atan2(y, x)
        return _replace_coordinates(X_features, X_features[..., 0] == CLUSTER_ELEMTYPE, eta, phi)
