"""Map each detector's track/cluster columns to one shared vocabulary of named features.
Inspired from Walrus (https://github.com/PolymathicAI/walrus, SpaceBagAdaptiveDVstrideEncoder),
the result is scaled by sqrt(vocabulary / present).
"""

import torch
import torch.nn as nn

from mlpf.conf import SOURCE_IDS, TRACK_CLUSTER_SOURCES, column_names

TRACK = 1
CLUSTER = 2
NUM_ELEMENT_TYPES = 3  # 0 padding, 1 track, 2 cluster

# The same quantity under different column names: {column name: feature}.
FEATURE_ALIASES = {
    "et": "pt",  # cluster transverse energy, the track pt slot
    "p": "energy",  # track momentum, the cluster energy slot
    "D0": "d0",
    "Z0": "z0",
}

# Per-dataset renames: IDEA clusters store Cherenkov/scintillation energies in the energy_ecal/energy_hcal columns.
FEATURE_OVERRIDES = {
    "idea": {CLUSTER: {"energy_ecal": "energy_cherenkov", "energy_hcal": "energy_scintillation"}},
}


def column_features(source, elemtype):
    """{column: feature} of one element type of a track/cluster dataset."""
    overrides = FEATURE_OVERRIDES.get(source, {}).get(elemtype, {})
    features = {}
    for column, name in enumerate(column_names(source, elemtype)):
        if column == 0 or name == "Null":
            continue
        features[column] = overrides.get(name, FEATURE_ALIASES.get(name, name))
    return features


SOURCE_FEATURE_COLUMNS = {source: {elemtype: column_features(source, elemtype) for elemtype in (TRACK, CLUSTER)} for source in TRACK_CLUSTER_SOURCES}

FEATURE_VOCABULARY = tuple(
    sorted({feature for by_type in SOURCE_FEATURE_COLUMNS.values() for columns in by_type.values() for feature in columns.values()})
)


def build_feature_gather_tables(input_dim):
    """Column read by each vocabulary slot, and that slot's scale, per (source id, element type).

    Returns two [num source ids * NUM_ELEMENT_TYPES, len(FEATURE_VOCABULARY)] tables, indexed by
    source_id * NUM_ELEMENT_TYPES + element type. A slot the dataset does not fill reads column 0 with
    scale 0; a filled slot has scale sqrt(vocabulary size / filled slots). Padding, unknown sources and
    sources without a track/cluster layout fill no slot.
    """
    vocabulary = {name: index for index, name in enumerate(FEATURE_VOCABULARY)}
    num_keys = (max(SOURCE_IDS.values()) + 1) * NUM_ELEMENT_TYPES
    column_of_slot = torch.zeros(num_keys, len(vocabulary), dtype=torch.long)
    slot_scale = torch.zeros(num_keys, len(vocabulary))
    for source, by_type in SOURCE_FEATURE_COLUMNS.items():
        for elemtype, columns in by_type.items():
            key = SOURCE_IDS[source] * NUM_ELEMENT_TYPES + elemtype
            slots = {vocabulary[feature]: column for column, feature in columns.items() if column < input_dim}
            if len(slots) != len([column for column in columns if column < input_dim]):
                raise ValueError(f"{source} element type {elemtype} has two columns for one feature: {columns}")
            for slot, column in slots.items():
                column_of_slot[key, slot] = column
                slot_scale[key, slot] = (len(vocabulary) / len(slots)) ** 0.5
    return column_of_slot, slot_scale


class FeatureBag(nn.Module):
    """Remap [B, N, input_dim] detector-specific features to [B, N, len(FEATURE_VOCABULARY)].

    Parameter-free: the learned projection is the first linear layer of the input
    encoder, which then sees each physical quantity in one fixed input slot.
    """

    def __init__(self, input_dim):
        super().__init__()
        column_of_slot, slot_scale = build_feature_gather_tables(input_dim)
        self.register_buffer("column_of_slot", column_of_slot, persistent=False)
        self.register_buffer("slot_scale", slot_scale, persistent=False)
        self.num_features = len(FEATURE_VOCABULARY)

    def forward(self, X_features, source_id):
        if source_id is None:
            raise ValueError("The feature bag layer needs each event's source_id")
        source = source_id.to(device=X_features.device, dtype=torch.long).view(-1, 1)
        key = source * NUM_ELEMENT_TYPES + X_features[..., 0].long().clamp(min=0, max=NUM_ELEMENT_TYPES - 1)
        # [B, N, vocabulary]: the column of each slot; slots the dataset does not fill read column 0.
        bag = torch.gather(X_features, -1, self.column_of_slot[key])
        # sqrt(vocabulary size / filled slots) on the filled slots, 0 on the others.
        scale = self.slot_scale[key].to(X_features.dtype)
        return bag * scale
