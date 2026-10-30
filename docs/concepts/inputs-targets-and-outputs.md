# Inputs, targets, and outputs

An event contains a variable number of detector elements. Track/cluster datasets start from locally reconstructed objects; hit datasets start from individual tracker and calorimeter cell-level hits. The [dataset catalog](../datasets/catalog.md) records the available representations and versions.

## Event contents

| Field | Meaning |
|---|---|
| `X` | One feature row per detector element: type, kinematic or geometric quantities, and detector-specific measurements |
| `ytarget` | Particle supervision aligned with the input rows |
| `ycand` | Reference reconstruction aligned with the inputs; the CLD/CLIC hit builders fill this with zeros |
| `genjets`, `targetjets`, `genmet` | Event-level reference quantities, where supplied by the dataset |

Feature order and particle labels are defined in [`mlpf/conf.py`](https://github.com/jpata/particleflow/blob/main/mlpf/conf.py) and the detector's TFDS builder. A shared array shape alone does not establish compatibility between representations. Use the matching dataset version, model recipe, and saved model configuration.

## Constructing particle targets

Postprocessing uses simulation relationships to associate particles with detector elements. Acceptance, available detector signals, merging, and detector-specific particle-type mappings determine the reconstructable target collection. Reference PF candidates are stored separately for comparison.

In the current Key4HEP hit representation, each retained particle has one representative hit carrying its full target row. Other associated hits carry its positive, event-local `particle_number`; zero denotes no assigned particle. For example:

| Input | Target class and kinematics | `particle_number` |
|---|---|---|
| Representative hit | Full particle target | 1 |
| Supporting hit | Zero | 1 |
| Unassigned hit | Zero | 0 |

The representative defines one particle for supervision, while the shared identifier supplies hit membership. The set-mode loader collects rows with nonzero target class into `ytarget_set` and checks uniqueness of nonzero particle identifiers. Hit-grouping supervision uses the identifiers on all associated hits.

Target construction depends on the detector and schema version. In particular, descendant-track ownership and the choice of a representative for a status-1 ancestor require care when secondary interactions change the track momentum or charge. See the [development priorities](../science/roadmap.md) and [dataset validation](../validation/dataset-validation.md) for the current limitations and checks.

## Batches and prediction coordinates

The loader pads events to form batches. The input mask is `X[..., 0] != 0`; padded rows are excluded from the standard particle loss. A valid input can have a zero target class, teaching the presence head whether that input should produce a particle.

Both output modes predict presence, particle identity, and five kinematic quantities. Their momentum coordinates differ:

| Quantity | Elementwise training | Set training | Decoded output |
|---|---|---|---|
| Transverse momentum | `log(target_pt / input_pt)` | `log(target_pt)` | `pt` |
| Direction | `eta`, `sin_phi`, `cos_phi` | Same | `eta`, with `phi = atan2(sin_phi, cos_phi)` |
| Energy | `log(target_energy / input_energy)` | `log(target_energy)` | `energy` |

Prediction decoding reverses the logarithmic transformations and zeroes inactive candidates. Dataset records also contain truth metadata such as charge and pileup labels. The current elementwise forward pass returns placeholder pileup logits; its trained heads cover presence, PID, and kinematics.

Implementation: [`PFDataset.py`](https://github.com/jpata/particleflow/blob/main/mlpf/model/PFDataset.py), [`MLPF.predict_particles`](https://github.com/jpata/particleflow/blob/main/mlpf/model/mlpf.py), and the [model architecture guide](model-architectures.md).
