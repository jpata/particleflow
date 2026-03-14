from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Dict, Any
import os
from enum import Enum
from mlpf.utils import resolve_path, load_spec, set_nested_dict


class Dataset(Enum):
    CMS = "cms"
    CLIC = "clic"
    CLD = "cld"
    CLIC_HITS = "clic_hits"


class ModelType(Enum):
    ATTENTION = "attention"
    GNN_LSH = "gnn_lsh"


class InputEncoding(Enum):
    JOINT = "joint"
    SPLIT = "split"


class LearnedRepresentationMode(Enum):
    LAST = "last"
    CONCAT = "concat"


class RegressionMode(Enum):
    DIRECT = "direct"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    DIRECT_ELEMTYPE = "direct-elemtype"
    DIRECT_ELEMTYPE_SPLIT = "direct-elemtype-split"
    LINEAR = "linear"
    LINEAR_ELEMTYPE = "linear-elemtype"


class Activation(Enum):
    ELU = "elu"
    RELU = "relu"
    RELU6 = "relu6"
    LEAKYRELU = "leakyrelu"
    GELU = "gelu"


class OptimizerType(Enum):
    ADAMW = "adamw"
    LAMB = "lamb"
    SGD = "sgd"


class LRSchedule(Enum):
    CONSTANT = "constant"
    ONECYCLE = "onecycle"
    COSINEDECAY = "cosinedecay"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


class AttentionType(Enum):
    MATH = "math"
    EFFICIENT = "efficient"
    FLASH = "flash"
    LINEAR = "linear"


# All possible PFElement types
# CMS classes from
# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
# 0 is placeholder for padded entries (generally only when batching events together)
ELEM_TYPES = {
    Dataset.CMS.value: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    Dataset.CLIC.value: [0, 1, 2],  # 1 - track, 2 - cluster
    Dataset.CLD.value: [0, 1, 2],  # 1 - track, 2 - cluster
}

# Some element types are defined, but do not exist in the dataset at all or should be excluded for physics reasons
ELEM_TYPES_NONZERO = {
    Dataset.CMS.value: [1, 4, 5, 6, 8, 9, 10, 11],
    Dataset.CLIC.value: [1, 2],
    Dataset.CLD.value: [1, 2],
}

CLASS_LABELS = {
    Dataset.CMS.value: [0, 211, 130, 1, 2, 22, 11, 13, 15],  # we never actually predict 15/taus (not there in targets)
    Dataset.CLIC.value: [0, 211, 130, 22, 11, 13],
    Dataset.CLD.value: [0, 211, 130, 22, 11, 13],
    Dataset.CLIC_HITS.value: [0, 211, 130, 22, 11, 13],
}

CLASS_NAMES_LATEX = {
    Dataset.CMS.value: ["none", "Charged Hadron", "Neutral Hadron", "HFEM", "HFHAD", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$", r"$\tau$"],
    Dataset.CLIC.value: ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    Dataset.CLD.value: ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    Dataset.CLIC_HITS.value: ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
}
CLASS_NAMES = {
    Dataset.CMS.value: ["none", "chhad", "nhad", "HFEM", "HFHAD", "gamma", "ele", "mu", "tau"],
    Dataset.CLIC.value: ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    Dataset.CLD.value: ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    Dataset.CLIC_HITS.value: ["none", "chhad", "nhad", "gamma", "ele", "mu"],
}
CLASS_NAMES_CAPITALIZED = {
    Dataset.CMS.value: ["none", "Charged hadron", "Neutral hadron", "HFEM", "HFHAD", "Photon", "Electron", "Muon", "Tau"],
    Dataset.CLIC.value: ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    Dataset.CLD.value: ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    Dataset.CLIC_HITS.value: ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
}

X_FEATURES = {
    Dataset.CMS.value: [
        "typ_idx",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "e",
        "layer",
        "depth",
        "charge",
        "trajpoint",
        "eta_ecal",
        "phi_ecal",
        "eta_hcal",
        "phi_hcal",
        "muon_dt_hits",
        "muon_csc_hits",
        "muon_type",
        "px",
        "py",
        "pz",
        "deltap",
        "sigmadeltap",
        "gsf_electronseed_trkorecal",
        "gsf_electronseed_dnn1",
        "gsf_electronseed_dnn2",
        "gsf_electronseed_dnn3",
        "gsf_electronseed_dnn4",
        "gsf_electronseed_dnn5",
        "num_hits",
        "cluster_flags",
        "corr_energy",
        "corr_energy_err",
        "vx",
        "vy",
        "vz",
        "pterror",
        "etaerror",
        "phierror",
        "lambd",
        "lambdaerror",
        "theta",
        "thetaerror",
        "time",
        "timeerror",
        "etaerror1",
        "etaerror2",
        "etaerror3",
        "etaerror4",
        "phierror1",
        "phierror2",
        "phierror3",
        "phierror4",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    ],
    Dataset.CLIC.value: [
        "type",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | iTheta",
        "radiusOfInnermostHit | energy_ecal",
        "tanLambda | energy_hcal",
        "D0 | energy_other",
        "omega | num_hits",
        "Z0 | sigma_x",
        "time | sigma_y",
        "Null | sigma_z",
    ],
    Dataset.CLD.value: [
        "type",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | iTheta",
        "radiusOfInnermostHit | energy_ecal",
        "tanLambda | energy_hcal",
        "D0 | energy_other",
        "omega | num_hits",
        "Z0 | sigma_x",
        "time | sigma_y",
        "Null | sigma_z",
    ],
    Dataset.CLIC_HITS.value: [
        "elemtype",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | time",
        "radiusOfInnermostHit | subdetector",
        "tanLambda | type",
        "D0 | Null",
        "omega | Null",
        "Z0 | Null",
        "time | Null",
    ],
}

JET_CONFIG = {
    Dataset.CMS.value: {
        "algo": "antikt_algorithm",
        "r": 0.4,
        "ptcut": 3.0,
        "match_dr": 0.1,
    },
    Dataset.CLIC.value: {
        "algo": "ee_genkt_algorithm",
        "r": 0.4,
        "p": -1.0,
        "ptcut": 5.0,
        "match_dr": 0.1,
    },
    Dataset.CLD.value: {
        "algo": "ee_genkt_algorithm",
        "r": 0.4,
        "p": -1.0,
        "ptcut": 5.0,
        "match_dr": 0.1,
    },
}

Y_FEATURES = [
    "PDG",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "ispu",
    "generatorStatus",
    "simulatorStatus",
    "gp_to_track",
    "gp_to_cluster",
    "jet_idx",
]


class GNNLSHConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv_type: ModelType = ModelType.GNN_LSH
    embedding_dim: int = 128
    width: int = 128
    num_convs: int = 2
    dropout_ff: float = 0.0
    activation: Activation = Activation.ELU
    layernorm: bool = True
    bin_size: int = 640
    max_num_bins: int = 200
    distance_dim: int = 128
    num_node_messages: int = 2
    ffn_dist_hidden_dim: int = 128
    ffn_dist_num_layers: int = 2


class AttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv_type: ModelType = ModelType.ATTENTION
    embedding_dim: int = 128
    width: int = 128
    num_convs: int = 2
    dropout_ff: float = 0.0
    activation: Activation = Activation.ELU
    layernorm: bool = True
    num_heads: int = 16
    head_dim: int = 16
    attention_type: AttentionType = AttentionType.FLASH
    dropout_conv_reg_mha: float = 0.0
    dropout_conv_reg_ff: float = 0.0
    dropout_conv_id_mha: float = 0.0
    dropout_conv_id_ff: float = 0.0
    use_pre_layernorm: bool = False
    use_simplified_attention: bool = False
    export_onnx_fused: bool = False
    save_attention: bool = False


class ModelArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ModelType
    input_encoding: InputEncoding = InputEncoding.SPLIT
    learned_representation_mode: LearnedRepresentationMode = LearnedRepresentationMode.LAST
    pt_mode: RegressionMode = RegressionMode.DIRECT_ELEMTYPE_SPLIT
    eta_mode: RegressionMode = RegressionMode.LINEAR
    sin_phi_mode: RegressionMode = RegressionMode.LINEAR
    cos_phi_mode: RegressionMode = RegressionMode.LINEAR
    energy_mode: RegressionMode = RegressionMode.DIRECT_ELEMTYPE_SPLIT
    trainable: str = "all"

    # Nested configs
    gnn_lsh: Optional[GNNLSHConfig] = None
    attention: Optional[AttentionConfig] = None


class DatasetSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    splits: List[str]
    batch_size: Optional[int] = None


class PhysicalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = 1
    samples: Dict[str, DatasetSample]


class TestDatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Optional[str] = None
    splits: List[str] = ["test"]
    batch_size: int = 1


class MLPFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Dataset
    data_dir: str
    model: ModelArchitectureConfig
    conv_type: ModelType

    batch_size: Optional[int] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Missing fields from spec
    backend: Optional[str] = None
    threads: Optional[int] = None
    gpu_type: Optional[str] = None
    mem_per_gpu_mb: Optional[int] = None
    slurm_partition: Optional[str] = None
    slurm_runtime: Optional[str] = None

    # Model dimensions (derived from dataset)
    input_dim: Optional[int] = None
    num_classes: Optional[int] = None
    elemtypes_nonzero: Optional[List[int]] = None

    # Training parameters
    num_steps: int = 100000
    patience: int = 10000
    checkpoint_freq: int = 10000
    val_freq: int = 10000
    num_workers: int = 8
    prefetch_factor: int = 2
    gpu_batch_multiplier: int = 1
    dtype: str = "float32"
    lr: float = 0.0001
    weight_decay: float = 0.01
    optimizer: OptimizerType = OptimizerType.ADAMW
    lr_schedule: LRSchedule = LRSchedule.COSINEDECAY
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict)
    pad_to_multiple_elements: Optional[int] = None

    # Flags
    train: bool = False
    test: bool = False
    make_plots: bool = False
    sort_data: bool = False
    load: Optional[str] = None
    relaxed_load: bool = True

    # Logging
    comet: bool = False
    comet_offline: bool = False
    comet_name: str = "particleflow"
    comet_step_freq: int = 10000

    raytune: Dict[str, Any] = Field(default_factory=dict)

    # Dataset specific
    train_dataset: Optional[Dict[str, Dict[str, PhysicalDataset]]] = None
    valid_dataset: Optional[Dict[str, Dict[str, PhysicalDataset]]] = None
    test_dataset: Dict[str, TestDatasetEntry] = Field(default_factory=dict)
    enabled_test_datasets: List[str] = Field(default_factory=list)

    # Sample limits
    ntrain: Optional[int] = None  # number of training events
    nvalid: Optional[int] = None  # number of validation events, ran at periodic intervals during training
    ntest: Optional[int] = None  # number of testing events, ran at the end of the training

    # Multi-GPU
    gpus: int = 0

    @model_validator(mode="after")
    def populate_defaults(self) -> "MLPFConfig":
        if self.dataset.value in X_FEATURES:
            if self.input_dim is None:
                self.input_dim = len(X_FEATURES[self.dataset.value])
            if self.num_classes is None:
                self.num_classes = len(CLASS_LABELS[self.dataset.value])
            if self.elemtypes_nonzero is None:
                self.elemtypes_nonzero = ELEM_TYPES_NONZERO[self.dataset.value]
        return self

    def flatten_config(self, prefix=""):
        """Flatten the nested configuration into a dot-separated path dictionary."""
        d = self.model_dump()
        items = {}

        def _flatten(d, prefix=""):
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, new_key)
                elif isinstance(v, Enum):
                    items[new_key] = v.value
                else:
                    items[new_key] = v

        _flatten(d, prefix)
        return items

    @staticmethod
    def from_spec(spec_file: str, model_name: str, production_name: str, args=None, extra_args=None):
        spec = load_spec(spec_file)

        if model_name not in spec["models"]:
            raise ValueError(f"Model {model_name} not found in spec")
        if production_name not in spec["productions"]:
            raise ValueError(f"Production {production_name} not found in spec")

        model_config_raw = spec["models"][model_name]
        prod_config_raw = spec["productions"][production_name]

        # Initialize config dict
        config_dict = {}

        # 1. Merge defaults
        if "defaults" in spec["models"]:
            for k, v in spec["models"]["defaults"].items():
                if isinstance(v, str):
                    v = resolve_path(v, spec)
                config_dict[k] = v

        # 2. Merge model config
        for k, v in model_config_raw.items():
            if k not in ["architecture", "train_datasets", "validation_datasets", "test_datasets"]:
                if isinstance(v, str):
                    v = resolve_path(v, spec)
                config_dict[k] = v

        if "hyperparameters" in model_config_raw:
            for k, v in model_config_raw["hyperparameters"].items():
                config_dict[k] = v

        # 3. Model Architecture
        config_dict["model"] = model_config_raw["architecture"]
        config_dict["conv_type"] = config_dict["model"]["type"]

        # 4. Dataset and Production
        config_dict["dataset"] = Dataset(model_config_raw.get("dataset", prod_config_raw.get("type")))
        workspace_dir = resolve_path(prod_config_raw["workspace_dir"], spec)
        config_dict["data_dir"] = os.path.join(workspace_dir, "tfds")

        # Set model dimensions
        ds_name = config_dict["dataset"].value
        config_dict["input_dim"] = len(X_FEATURES[ds_name])
        config_dict["num_classes"] = len(CLASS_LABELS[ds_name])
        config_dict["elemtypes_nonzero"] = ELEM_TYPES_NONZERO[ds_name]

        # Helper for datasets
        def build_dataset_config_dict(dataset_input):
            ds_config = {}
            ds_config[ds_name] = {}
            for phys_key, phys_val in dataset_input.items():
                ds_config[ds_name][phys_key] = {
                    "batch_size": phys_val.get("batch_size", config_dict.get("batch_size", 1)),
                    "samples": {},
                }
                target_dict = ds_config[ds_name][phys_key]["samples"]
                for ds_item in phys_val["samples"]:
                    name = ds_item["name"]
                    entry = {"version": ds_item.get("version"), "splits": ds_item.get("splits")}
                    if "batch_size" in ds_item:
                        entry["batch_size"] = ds_item["batch_size"]
                    target_dict[name] = entry
            return ds_config

        if "train_datasets" in model_config_raw:
            config_dict["train_dataset"] = build_dataset_config_dict(model_config_raw["train_datasets"])
        if "validation_datasets" in model_config_raw:
            config_dict["valid_dataset"] = build_dataset_config_dict(model_config_raw["validation_datasets"])
        if "test_datasets" in model_config_raw:
            config_dict["test_dataset"] = {}
            for ds_item in model_config_raw.get("test_datasets", []):
                name = ds_item["name"]
                config_dict["test_dataset"][name] = {
                    "version": ds_item.get("version"),
                    "splits": ds_item.get("splits", ["test"]),
                    "batch_size": ds_item.get("batch_size", 1),
                }

        # 5. Apply Argparse overrides
        if args:
            for arg in vars(args):
                val = getattr(args, arg)
                if val is not None:
                    # Direct override if key exists in config_dict
                    if arg in config_dict:
                        config_dict[arg] = val

            # Action flags
            for flag in ["train", "test", "make_plots", "gpus", "load"]:
                if hasattr(args, flag) and getattr(args, flag) is not None:
                    config_dict[flag] = getattr(args, flag)

            # Special mapping cases (convenience flags)
            if hasattr(args, "attention_type") and args.attention_type is not None:
                set_nested_dict(config_dict, "model.attention.attention_type", args.attention_type)

            if hasattr(args, "num_convs") and args.num_convs is not None:
                for m in ["gnn_lsh", "attention"]:
                    if m in config_dict["model"]:
                        set_nested_dict(config_dict, f"model.{m}.num_convs", args.num_convs)

        # 6. Apply Dot-notation overrides (extra_args)
        if extra_args:
            from mlpf.utils import parse_extra_args

            overrides = parse_extra_args(extra_args)
            for key, value in overrides.items():
                set_nested_dict(config_dict, key, value)

            # Check for leftover extra arguments that could not be parsed as overrides
            i = 0
            while i < len(extra_args):
                arg = extra_args[i]
                if arg.startswith("--"):
                    if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--") and "=" not in extra_args[i + 1]:
                        i += 2
                    else:
                        i += 1
                elif "=" in arg:
                    i += 1
                else:
                    raise ValueError(f"Could not parse extra argument: {arg}")

        # 7. Pipeline Overrides
        if args and hasattr(args, "pipeline") and args.pipeline:
            # Replicate pipeline-specific overrides
            if "gnn_lsh" not in config_dict["model"]:
                config_dict["model"]["gnn_lsh"] = {}
            config_dict["model"]["gnn_lsh"]["num_convs"] = 1
            config_dict["model"]["gnn_lsh"]["width"] = 32
            config_dict["model"]["gnn_lsh"]["embedding_dim"] = 32

            if "attention" not in config_dict["model"]:
                config_dict["model"]["attention"] = {}
            config_dict["model"]["attention"]["num_convs"] = 1
            config_dict["model"]["attention"]["num_heads"] = 2
            config_dict["model"]["attention"]["head_dim"] = 2

            if ds_name == "cms":
                for ds in ["train_dataset", "valid_dataset"]:
                    if ds in config_dict:
                        config_dict[ds][ds_name] = {
                            "physical_pu": {
                                "batch_size": config_dict[ds][ds_name]["physical_pu"]["batch_size"],
                                "samples": {"cms_pf_ttbar": {"splits": ["10"], "version": "3.0.0"}},
                            }
                        }
                if "cms_pf_ttbar" in config_dict["test_dataset"]:
                    config_dict["test_dataset"] = {"cms_pf_ttbar": config_dict["test_dataset"]["cms_pf_ttbar"]}
                    config_dict["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]

        # 8. Post-override adjustments
        if "test_dataset" in config_dict:
            config_dict["enabled_test_datasets"] = list(config_dict["test_dataset"].keys())
        if args and hasattr(args, "test_datasets") and args.test_datasets:
            config_dict["enabled_test_datasets"] = args.test_datasets

        # 9. Validate with Pydantic
        return MLPFConfig.model_validate(config_dict)
