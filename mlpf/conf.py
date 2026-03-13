from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Dict, Union, Any, Literal
import yaml
import os
from mlpf.utils import resolve_path, load_spec, set_nested_dict


class GNNLSHConfig(BaseModel):
    conv_type: Literal["gnn_lsh"] = "gnn_lsh"
    embedding_dim: int = 128
    width: int = 128
    num_convs: int = 2
    dropout_ff: float = 0.0
    activation: str = "elu"
    layernorm: bool = True
    bin_size: int = 640
    max_num_bins: int = 200
    distance_dim: int = 128
    num_node_messages: int = 2
    ffn_dist_hidden_dim: int = 128
    ffn_dist_num_layers: int = 2


class AttentionConfig(BaseModel):
    conv_type: Literal["attention"] = "attention"
    embedding_dim: int = 128
    width: int = 128
    num_convs: int = 2
    dropout_ff: float = 0.0
    activation: str = "elu"
    layernorm: bool = True
    num_heads: int = 16
    head_dim: int = 16
    attention_type: str = "flash"
    dropout_conv_reg_mha: float = 0.0
    dropout_conv_reg_ff: float = 0.0
    dropout_conv_id_mha: float = 0.0
    dropout_conv_id_ff: float = 0.0
    use_pre_layernorm: bool = False
    use_simplified_attention: bool = False
    export_onnx_fused: bool = False
    save_attention: bool = False


class ModelArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["attention", "gnn_lsh"]
    input_encoding: str = "split"
    learned_representation_mode: str = "last"
    pt_mode: str = "direct-elemtype-split"
    eta_mode: str = "linear"
    sin_phi_mode: str = "linear"
    cos_phi_mode: str = "linear"
    energy_mode: str = "direct-elemtype-split"
    trainable: str = "all"

    # Nested configs
    gnn_lsh: Optional[GNNLSHConfig] = None
    attention: Optional[AttentionConfig] = None


class DatasetSample(BaseModel):
    version: str
    splits: List[str]
    batch_size: Optional[int] = None


class PhysicalDataset(BaseModel):
    batch_size: int = 1
    samples: Dict[str, DatasetSample]


class TestDatasetEntry(BaseModel):
    version: Optional[str] = None
    splits: List[str] = ["test"]
    batch_size: int = 1


class MLPFConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str
    data_dir: str
    model: ModelArchitectureConfig
    conv_type: str

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
    optimizer: str = "adamw"
    lr_schedule: str = "cosinedecay"
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict)

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

    # Dataset specific
    train_dataset: Optional[Dict[str, Dict[str, PhysicalDataset]]] = None
    valid_dataset: Optional[Dict[str, Dict[str, PhysicalDataset]]] = None
    test_dataset: Dict[str, TestDatasetEntry] = Field(default_factory=dict)
    enabled_test_datasets: List[str] = Field(default_factory=list)

    # Sample limits
    ntrain: Optional[int] = None
    nvalid: Optional[int] = None
    ntest: Optional[int] = None

    # Multi-GPU
    gpus: int = 0

    def flatten_config(self, prefix=""):
        """Flatten the nested configuration into a dot-separated path dictionary."""
        d = self.model_dump()
        items = {}

        def _flatten(d, prefix=""):
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, new_key)
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
        config_dict["dataset"] = model_config_raw.get("dataset", prod_config_raw.get("type"))
        workspace_dir = resolve_path(prod_config_raw["workspace_dir"], spec)
        config_dict["data_dir"] = os.path.join(workspace_dir, "tfds")

        # Helper for datasets
        def build_dataset_config_dict(dataset_input):
            ds_config = {}
            ds_config[config_dict["dataset"]] = {}
            for phys_key, phys_val in dataset_input.items():
                ds_config[config_dict["dataset"]][phys_key] = {
                    "batch_size": phys_val.get("batch_size", config_dict.get("batch_size", 1)),
                    "samples": {},
                }
                target_dict = ds_config[config_dict["dataset"]][phys_key]["samples"]
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

            if config_dict["dataset"] == "cms":
                for ds in ["train_dataset", "valid_dataset"]:
                    if ds in config_dict:
                        config_dict[ds]["cms"] = {
                            "physical_pu": {
                                "batch_size": config_dict[ds]["cms"]["physical_pu"]["batch_size"],
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
