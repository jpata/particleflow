import sys

import os
import time
import gc
import json
import platform
import subprocess
import psutil

# Ensure mlpf is in the path
sys.path.insert(0, os.getcwd())

import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplhep
import boost_histogram as bh
import torch
import onnxruntime as rt
import awkward
import vector
import fastjet

# The optimized direct bucket gather path bakes sequence lengths into the
# legacy TorchScript ONNX export. Use the export-safe path for validation.
os.environ.setdefault("HEPTV2_DIRECT_BUCKET_GATHER", "0")

print("onnxruntime", rt.__path__)
from mlpf.model.mlpf import MLPF  # noqa: E402
from mlpf.model.onnx_attention import export_mlpf, validate_padding_mask  # noqa: E402
from mlpf.model.onnx_runtime import create_session, finish_profile  # noqa: E402
from mlpf.conf import MLPFConfig, ModelType, AttentionType  # noqa: E402
from mlpf.model.utils import unpack_predictions  # noqa: E402
from mlpf.plotting.plot_utils import ELEM_NAMES_CMS, CLASS_NAMES_CMS, CLASS_NAMES_CLIC  # noqa: E402

print("Imports finished.")


def get_labels(dataset):
    if "cms" in dataset:
        return ELEM_NAMES_CMS, CLASS_NAMES_CMS, [1, 4, 5, 6, 8, 9], "cms"
    elif "cld" in dataset:
        return ["NONE", "TRACK", "CLUSTER"], CLASS_NAMES_CLIC, [1, 2], "cld"
    elif "clic" in dataset:
        return ["NONE", "TRACK", "CLUSTER"], CLASS_NAMES_CLIC, [1, 2], "clic"
    else:
        return ELEM_NAMES_CMS, CLASS_NAMES_CMS, [1, 4, 5, 6, 8, 9], "cms"


def make_mlpf_config(model_kwargs: MLPFConfig, **overrides):
    config = model_kwargs.model_copy(deep=True)
    # Apply overrides
    for k, v in overrides.items():
        if k in ["export_onnx_fused", "save_attention", "attention_type", "use_jagged_attention", "use_flash_attn_varlen"]:
            if config.model.type == ModelType.ATTENTION:
                setattr(config.model.attention, k, v)
        elif hasattr(config, k):
            setattr(config, k, v)
        elif hasattr(config.model, k):
            setattr(config.model, k, v)
    return config


# Updated configurations
SUPPORTED_CONFIGS = [
    # Attention-based models
    "ONNX_ATTN_OP_FP32",
    "ONNX_ATTN_FLASH_FP16",
    "ONNX_ATTN_FLASH_FP32_FP16",
    "ONNX_ATTN_MATH_FP16",
    "ONNX_ATTN_MATH_FP32_FP16",
    "ONNX_ATTN_MATH_FP32",
    "PT_ATTN_FLASH_FP16",
    "PT_ATTN_MATH_FP16",
    "PT_ATTN_MATH_FP32",
    # HEPT model
    "PT_HEPT_FP32",
    "ONNX_HEPT_FP32",
    # HEPTv2 model
    "PT_HEPTV2_FP32",
    "ONNX_HEPTV2_FP32",
    # LitePT model
    "PT_LITEPT_FP32",
    # GNN-LSH model
    "PT_GNNLSH_FP32",
    "ONNX_GNNLSH_FP32",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export MLPF model to ONNX and validate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pth file")
    parser.add_argument("--model-kwargs", type=str, required=True, help="Path to the model_kwargs.pkl file")
    parser.add_argument("--dataset", type=str, default="cms_pf_ttbar", help="TFDS dataset name")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory for TFDS datasets")
    parser.add_argument("--outdir", type=str, default="./onnx_validation", help="Output directory for ONNX and plots")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events for validation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--ort-provider", choices=["cpu", "cuda", "migraphx"], default=None)
    parser.add_argument("--require-full-gpu", action="store_true", help="Reject ONNX CPU compute fallback and retain provider profiles")
    parser.add_argument("--pad-bin-size", type=int, default=0, help="Pad events to the nearest multiple of this size")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=SUPPORTED_CONFIGS,
        default=SUPPORTED_CONFIGS,
        help="List of configurations to run",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Events per inference batch (right-padded independently)")
    parser.add_argument("--save-inputs", action="store_true", help="Save NPZ batches for standalone AMD/CPU/CUDA replay")
    parser.add_argument("--profile", action="store_true", help="Save ONNX execution-provider profiles")
    parser.add_argument("--compile", action="store_true", help="Compile PyTorch models before validation")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of CPU threads to use")
    return parser.parse_args()


def particles_to_jets(pred, mask):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    ypred = unpack_predictions(pred)
    for k, v in ypred.items():
        ypred[k] = v[mask].detach().cpu().float().contiguous().numpy()

    counts = torch.sum(mask, axis=1).cpu().numpy()
    clsid = awkward.unflatten(ypred["cls_id"], counts)
    msk = clsid != 0
    p4 = awkward.unflatten(ypred["p4"], counts)

    vec = vector.awk(
        awkward.zip(
            {
                "pt": p4[msk][:, :, 0],
                "eta": p4[msk][:, :, 1],
                "phi": p4[msk][:, :, 2],
                "e": p4[msk][:, :, 3],
            }
        )
    )
    cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
    jets = cluster.inclusive_jets(min_pt=3)
    return jets


def sum_overflow_into_last_bin(all_values):
    values = all_values[1:-1]
    values[-1] = values[-1] + all_values[-1]
    values[0] = values[0] + all_values[0]
    return values


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    h1[:] = sum_overflow_into_last_bin(h1.values(flow=True)[:])
    return h1


@torch.no_grad()
def calculate_delta_r_numba(particles):
    from numba import njit

    @njit
    def _calculate_delta_r(particles):
        n_particles = particles.shape[0]
        delta_r_matrix = np.empty((n_particles, n_particles), dtype=np.float64)
        phis = np.empty(n_particles, dtype=np.float64)
        for i in range(n_particles):
            phis[i] = np.arctan2(particles[i, 1], particles[i, 2])
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    delta_r_matrix[i, j] = 0.0
                    continue
                eta1 = particles[i, 0]
                phi1 = phis[i]
                eta2 = particles[j, 0]
                phi2 = phis[j]
                delta_eta = eta1 - eta2
                delta_phi = phi1 - phi2
                delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
                delta_r_matrix[i, j] = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_r_matrix

    return _calculate_delta_r(particles)


def get_cpu_info():
    try:
        return subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().split(":")[1].strip()
    except Exception:
        return platform.processor() or "Unknown CPU"


def get_gpu_info():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown GPU"


def main():
    print("Starting MLPF ONNX validation script...")
    args = parse_args()
    if args.batch_size < 1 or args.num_events < 1:
        raise ValueError("--batch-size and --num-events must be positive")
    args.ort_provider = args.ort_provider or args.device
    if args.require_full_gpu and args.ort_provider == "cpu":
        raise ValueError("--require-full-gpu requires a GPU ONNX provider")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU PyTorch is unavailable; --device cuda also names ROCm devices")

    # Set seeds for reproducibility during model initialization and export
    torch.manual_seed(42)
    np.random.seed(42)

    torch.set_num_threads(args.num_threads)
    os.makedirs(args.outdir, exist_ok=True)
    mplhep.style.use("CMS")

    path_math_fp32 = os.path.join(args.outdir, "model_math_fp32.onnx")
    path_math_fp16 = os.path.join(args.outdir, "model_math_fp16.onnx")
    path_math_fp32_fp16 = os.path.join(args.outdir, "model_math_fp32_fp16.onnx")
    path_fused_fp32 = os.path.join(args.outdir, "model_fused_fp32.onnx")
    path_fused_fp32_fp16 = os.path.join(args.outdir, "model_fused_fp32_fp16.onnx")
    path_fused_fp16 = os.path.join(args.outdir, "model_fused_fp16.onnx")

    elem_names, class_names, typs, exp_name = get_labels(args.dataset)

    # Load model configuration
    print(f"Loading model configuration from {args.model_kwargs}...")
    with open(args.model_kwargs, "rb") as f:
        model_kwargs_raw = pkl.load(f)
        if isinstance(model_kwargs_raw, dict):
            model_kwargs = MLPFConfig.model_validate(model_kwargs_raw)
        else:
            # if it was already an MLPFConfig object (from a previous version),
            # re-validate it to ensure Enums are correct
            model_kwargs = MLPFConfig.model_validate(model_kwargs_raw.model_dump())

    print(model_kwargs)

    # Load model weights
    print(f"Loading model weights from {args.checkpoint}...")
    model_state = torch.load(args.checkpoint, map_location=torch.device("cpu"), weights_only=True)
    # Training attaches loss calibration buffers that inference models do not have.
    # Remove only those buffers so all model weights are still loaded strictly.
    state_dict = model_state["model_state_dict"]
    for key in list(state_dict):
        if key.startswith("task_loss_weighter."):
            del state_dict[key]

    # Load step losses if available
    print("Attempting to load step losses...")
    train_loss = None
    valid_loss = None
    try:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        experiment_dir = os.path.dirname(checkpoint_dir)
        step_num = os.path.basename(args.checkpoint).split("-")[1].split(".")[0]
        history_path = os.path.join(experiment_dir, "history", f"step_{step_num}.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history_data = json.load(f)
                train_loss = history_data.get("train", {}).get("Total")
                valid_loss = history_data.get("valid", {}).get("Total")
                print(f"Loaded losses from {history_path}: train={train_loss}, valid={valid_loss}")
        else:
            print(f"No history file found at {history_path}")
    except Exception as e:
        print(f"Could not load step losses: {e}")

    input_dim = model_kwargs.input_dim

    # For HEPT and GNN-LSH models, we MUST pad to block_size externally for ONNX
    pad_size = args.pad_bin_size
    if model_kwargs.pad_to_multiple_elements:
        pad_size = max(pad_size, model_kwargs.pad_to_multiple_elements)
    if model_kwargs.model.type == ModelType.HEPT:
        pad_size = max(pad_size, model_kwargs.model.hept.block_size)
    elif model_kwargs.model.type == ModelType.HEPTV2:
        pad_size = max(pad_size, model_kwargs.model.heptv2.block_size)
    elif model_kwargs.model.type == ModelType.GNNLSH:
        pad_size = max(pad_size, model_kwargs.model.gnnlsh.bin_size)

    configs = args.configs
    # Auto-select configs based on model type if not specified
    if not args.configs or set(args.configs) == set(SUPPORTED_CONFIGS):
        if model_kwargs.model.type == ModelType.ATTENTION:
            configs = [
                "PT_ATTN_MATH_FP32",
                "PT_ATTN_MATH_FP16",
                "PT_ATTN_FLASH_FP16",
                "ONNX_ATTN_MATH_FP32",
                "ONNX_ATTN_MATH_FP16",
                "ONNX_ATTN_MATH_FP32_FP16",
                "ONNX_ATTN_OP_FP32",
                "ONNX_ATTN_FLASH_FP32_FP16",
                "ONNX_ATTN_FLASH_FP16",
            ]
        elif model_kwargs.model.type == ModelType.HEPT:
            configs = ["PT_HEPT_FP32", "ONNX_HEPT_FP32"]
        elif model_kwargs.model.type == ModelType.HEPTV2:
            configs = ["PT_HEPTV2_FP32", "ONNX_HEPTV2_FP32"]
        elif model_kwargs.model.type == ModelType.LITEPT:
            configs = ["PT_LITEPT_FP32"]
        elif model_kwargs.model.type == ModelType.GNNLSH:
            configs = ["PT_GNNLSH_FP32", "ONNX_GNNLSH_FP32"]

    if args.device == "cpu":
        if args.configs != SUPPORTED_CONFIGS and any("FLASH" in cfg for cfg in configs):
            raise ValueError("FLASH configurations require --device cuda; use ONNX_ATTN_OP_FP32 for CPU Attention-24")
        configs = [c for c in configs if "LITEPT" not in c and "FLASH" not in c]
        if not configs:
            print("LitePT is not supported on CPU. Skipping.")
            return

    baseline = configs[0]
    # Reorder configs to run baseline first for comparison
    configs = [baseline] + [c for c in configs if c != baseline]

    # Export with B>1 and unequal lengths, regardless of the inference batch size.
    export_dummy_size = max(32, pad_size)
    dummy_features = torch.randn(2, export_dummy_size, input_dim)
    dummy_features[..., 0] = model_kwargs.elemtypes_nonzero[0]
    dummy_features[..., 1] = dummy_features[..., 1].exp()
    dummy_features[..., 5] = dummy_features[..., 5].exp()
    dummy_mask = torch.ones(2, export_dummy_size)
    dummy_features[1, export_dummy_size // 2 :] = 0
    dummy_mask[1, export_dummy_size // 2 :] = 0
    paths = {
        "ONNX_ATTN_MATH_FP32": path_math_fp32,
        "ONNX_ATTN_MATH_FP16": path_math_fp16,
        "ONNX_ATTN_MATH_FP32_FP16": path_math_fp32_fp16,
        "ONNX_ATTN_OP_FP32": path_fused_fp32,
        "ONNX_ATTN_FLASH_FP32_FP16": path_fused_fp32_fp16,
        "ONNX_ATTN_FLASH_FP16": path_fused_fp16,
        "ONNX_HEPT_FP32": path_math_fp32,
        "ONNX_HEPTV2_FP32": path_math_fp32,
        "ONNX_GNNLSH_FP32": path_math_fp32,
    }
    for cfg in configs:
        if cfg not in paths:
            continue
        standard_attention = cfg in ("ONNX_ATTN_OP_FP32", "ONNX_ATTN_FLASH_FP32_FP16", "ONNX_ATTN_FLASH_FP16")
        export_config = make_mlpf_config(
            model_kwargs,
            attention_type=AttentionType.MATH,
            export_onnx_fused=standard_attention,
            use_jagged_attention=False,
            use_flash_attn_varlen=False,
            save_attention=False,
        )
        model_export = MLPF(export_config).eval()
        model_export.load_state_dict(model_state["model_state_dict"], strict=True)
        full_half = cfg in ("ONNX_ATTN_MATH_FP16", "ONNX_ATTN_FLASH_FP16")
        if full_half:
            model_export = model_export.half()
        export_mlpf(
            model_export,
            dummy_features.half() if full_half else dummy_features,
            dummy_mask.half() if full_half else dummy_mask,
            paths[cfg],
            attention_format="attention24" if standard_attention else "decomposed",
            attention_dtype=torch.float16 if cfg in ("ONNX_ATTN_FLASH_FP32_FP16", "ONNX_ATTN_MATH_FP32_FP16") else None,
        )
        del model_export

    # Validation Setup
    print(f"Loading dataset {args.dataset} from {args.data_dir}...")
    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    ds = builder.as_data_source(split="train")
    args.num_events = min(args.num_events, len(ds))

    results = {
        cfg: {
            "id": [],
            "momentum": [],
            "pu": [],
            "total_err": 0.0,
            "num_elems": 0,
            "num_invalid": 0,
            "runtime": [],
            "memory_max": [],
            "event_size": [],
            "jets_pt": [],
            "runs": [],
        }
        for cfg in configs
    }

    all_event_sizes = []
    for i in range(args.num_events):
        all_event_sizes.append(ds[i]["X"].shape[0])

    baseline_predictions = {}
    process = psutil.Process()

    provider_profiles = {}

    for cfg in configs:
        print(f"Running validation for {cfg}...")
        if "ONNX_ATTN_FLASH" in cfg:
            os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"

        # 1. Initialize model/session dynamically for this configuration
        model = None
        sess = None

        if cfg in ["PT_ATTN_MATH_FP32", "PT_ATTN_MATH_FP16"]:
            print("Initializing PyTorch ATTN_MATH model...")
            model_kwargs_math = model_kwargs.model_copy(deep=True)
            model = MLPF(
                config=make_mlpf_config(
                    model_kwargs_math,
                    attention_type=AttentionType.MATH,
                    use_jagged_attention=False,
                    use_flash_attn_varlen=False,
                    export_onnx_fused=False,
                    save_attention=False,
                ),
            )
            model.eval()
            model.load_state_dict(model_state["model_state_dict"], strict=True)
            model = model.to(device=args.device)

        elif cfg == "PT_ATTN_FLASH_FP16":
            print("Initializing PyTorch Flash model...")
            model_kwargs_flash = model_kwargs.model_copy(deep=True)
            model = MLPF(
                config=make_mlpf_config(
                    model_kwargs_flash,
                    attention_type=AttentionType.FLASH,
                    use_jagged_attention=True,
                    use_flash_attn_varlen=True,
                    export_onnx_fused=False,
                    save_attention=False,
                ),
            )
            model.eval()
            model.load_state_dict(model_state["model_state_dict"], strict=True)
            model = model.to(device=args.device)

        elif cfg in ["PT_HEPT_FP32", "PT_HEPTV2_FP32"]:
            print(f"Initializing PyTorch {cfg.split('_')[1]} model...")
            model = MLPF(config=model_kwargs)
            model.eval()
            model.load_state_dict(model_state["model_state_dict"], strict=True)
            model = model.to(device=args.device)

        elif cfg == "PT_LITEPT_FP32":
            print("Initializing PyTorch LitePT model...")
            model = MLPF(config=model_kwargs)
            model.eval()
            model.load_state_dict(model_state["model_state_dict"], strict=True)
            model = model.to(device=args.device)

        elif cfg == "PT_GNNLSH_FP32":
            print("Initializing PyTorch GNNLSH model...")
            model = MLPF(config=model_kwargs)
            model.eval()
            model.load_state_dict(model_state["model_state_dict"], strict=True)
            model = model.to(device=args.device)

        elif cfg in paths:
            sess = create_session(
                paths[cfg],
                args.ort_provider,
                threads=args.num_threads,
                profile_prefix=os.path.join(args.outdir, cfg) if args.profile or args.require_full_gpu else None,
                require_full_gpu=args.require_full_gpu,
                flash="ONNX_ATTN_FLASH" in cfg,
            )

        if model is not None and args.compile:
            if model_kwargs.model.type == ModelType.LITEPT:
                print("Skipping PyTorch compilation for LitePT model...")
            else:
                print("Compiling PyTorch model...")
                model = torch.compile(model)

        # 2. Warm up this specific model/session using a small sequence length to save CPU time and memory
        print(f"Performing warmup for {cfg}...")
        warmup_seq_len = 256
        if pad_size > 0:
            if warmup_seq_len % pad_size != 0:
                warmup_seq_len = ((warmup_seq_len + pad_size - 1) // pad_size) * pad_size

        X_warmup = torch.randn(args.batch_size, warmup_seq_len, input_dim).float().to(args.device)
        mask_warmup = torch.ones(args.batch_size, warmup_seq_len, dtype=torch.bool, device=args.device)

        X_warmup_np = X_warmup.cpu().float().numpy()
        mask_f_warmup = mask_warmup.float().cpu().numpy()
        X_warmup_np_fp16 = X_warmup_np.astype(np.float16)
        mask_f_warmup_fp16 = mask_f_warmup.astype(np.float16)

        for _ in range(2):
            if model is not None:
                with torch.no_grad():
                    if cfg == "PT_ATTN_MATH_FP32":
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            _ = model(X_warmup, mask_warmup)
                    elif cfg == "PT_ATTN_MATH_FP16":
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                                _ = model(X_warmup, mask_warmup)
                    elif cfg == "PT_ATTN_FLASH_FP16":
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            _ = model(X_warmup, mask_warmup)
                    else:
                        _ = model(X_warmup, mask_warmup)
            elif sess is not None:
                if cfg in [
                    "ONNX_ATTN_MATH_FP32",
                    "ONNX_HEPT_FP32",
                    "ONNX_HEPTV2_FP32",
                    "ONNX_GNNLSH_FP32",
                    "ONNX_ATTN_OP_FP32",
                    "ONNX_ATTN_FLASH_FP32_FP16",
                    "ONNX_ATTN_MATH_FP32_FP16",
                ]:
                    _ = sess.run(None, {"Xfeat_normed": X_warmup_np, "mask": mask_f_warmup})
                elif cfg in ["ONNX_ATTN_MATH_FP16", "ONNX_ATTN_FLASH_FP16"]:
                    _ = sess.run(None, {"Xfeat_normed": X_warmup_np_fp16, "mask": mask_f_warmup_fp16})
        print(f"Warmup for {cfg} complete. Starting validation.")

        # 3. Run validation for this configuration
        for i in tqdm(range(0, args.num_events, args.batch_size)):
            event_indices = list(range(i, min(i + args.batch_size, args.num_events)))
            events = [torch.tensor(ds[j]["X"], dtype=torch.float32) for j in event_indices]
            # Match TFDSDataSource's feature-width contract for mixed schema versions.
            if any(event.shape[1] > input_dim for event in events):
                raise ValueError(f"Dataset feature width exceeds model input_dim={input_dim}")
            events = [torch.nn.functional.pad(event, (0, input_dim - event.shape[1])) for event in events]
            if model_kwargs.sort_data:
                events = [event[torch.argsort(event[:, 1], descending=True)] for event in events]
            # Drop pre-existing zero padding before forming a new right-padded batch.
            events = [event[event[:, 0] != 0] for event in events]
            lengths = [len(event) for event in events]
            if not all(lengths):
                raise ValueError(f"Empty event in batch {event_indices}; remove empty events before inference")
            num_elements = sum(lengths)
            X_features_padded = torch.nn.utils.rnn.pad_sequence(events, batch_first=True).to(args.device)
            padded_length = X_features_padded.shape[1]
            if pad_size > 0:
                padded_length = ((padded_length + pad_size - 1) // pad_size) * pad_size
                X_features_padded = torch.nn.functional.pad(X_features_padded, (0, 0, 0, padded_length - X_features_padded.shape[1]))
            X_features_padded = X_features_padded.clamp(-60000, 60000).contiguous()
            mask = torch.arange(padded_length, device=args.device)[None, :] < torch.tensor(lengths, device=args.device)[:, None]
            validate_padding_mask(mask)
            mask_f = mask.float()
            m_cpu = mask.cpu()

            # Pre-convert to numpy for ONNX
            X_features_np = X_features_padded.cpu().float().numpy()
            mask_f_np = mask_f.cpu().numpy()
            if args.save_inputs and cfg == baseline:
                np.savez(os.path.join(args.outdir, f"inputs_{i:06d}.npz"), Xfeat_normed=X_features_np, mask=mask_f_np)

            try:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                if cfg == "PT_ATTN_MATH_FP32":
                    with torch.no_grad():
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            pred = model(X_features_padded, mask)
                elif cfg == "PT_ATTN_MATH_FP16":
                    with torch.no_grad():
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                                pred = model(X_features_padded, mask)
                elif cfg == "PT_ATTN_FLASH_FP16":
                    with torch.no_grad():
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            pred = model(X_features_padded, mask)
                elif cfg in ["PT_HEPT_FP32", "PT_HEPTV2_FP32", "PT_LITEPT_FP32", "PT_GNNLSH_FP32"]:
                    with torch.no_grad():
                        pred = model(X_features_padded, mask)
                elif cfg in ["ONNX_ATTN_MATH_FP32", "ONNX_HEPT_FP32", "ONNX_HEPTV2_FP32", "ONNX_GNNLSH_FP32"]:
                    pred = sess.run(None, {"Xfeat_normed": X_features_np, "mask": mask_f_np})
                    pred = tuple(torch.tensor(p) for p in pred)
                elif cfg == "ONNX_ATTN_OP_FP32":
                    pred = sess.run(None, {"Xfeat_normed": X_features_np, "mask": mask_f_np})
                    pred = tuple(torch.tensor(p).float() for p in pred)
                elif cfg == "ONNX_ATTN_MATH_FP16":
                    X_features_np_fp16 = X_features_np.astype(np.float16)
                    mask_f_np_fp16 = mask_f_np.astype(np.float16)
                    pred = sess.run(None, {"Xfeat_normed": X_features_np_fp16, "mask": mask_f_np_fp16})
                    pred = tuple(torch.tensor(p).float() for p in pred)
                elif cfg in ("ONNX_ATTN_FLASH_FP32_FP16", "ONNX_ATTN_MATH_FP32_FP16"):
                    pred = sess.run(None, {"Xfeat_normed": X_features_np, "mask": mask_f_np})
                    pred = tuple(torch.tensor(p).float() for p in pred)
                elif cfg == "ONNX_ATTN_FLASH_FP16":
                    X_features_np_fp16 = X_features_np.astype(np.float16)
                    mask_f_np_fp16 = mask_f_np.astype(np.float16)
                    pred = sess.run(None, {"Xfeat_normed": X_features_np_fp16, "mask": mask_f_np_fp16})
                    pred = tuple(torch.tensor(p).float() for p in pred)

                if args.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                if args.device == "cuda":
                    mem_max = torch.cuda.max_memory_allocated() / (1024**2) if model is not None else None  # ORT memory is not sampled
                else:
                    mem_max = process.memory_info().rss / (1024**2)  # In MiB

                if "PT" in cfg:
                    pred = tuple(p.cpu().float() for p in pred)

                if cfg == baseline:
                    baseline_predictions[i] = (pred, m_cpu)

                results[cfg]["runtime"].append(t1 - t0)
                if mem_max is not None:
                    results[cfg]["memory_max"].append(mem_max)
                results[cfg]["event_size"].append(num_elements)
                results[cfg]["runs"].append(
                    {
                        "event_idx": i,
                        "event_indices": event_indices,
                        "batch_size": len(lengths),
                        "lengths": lengths,
                        "padded_length": padded_length,
                        "size": num_elements,
                        "runtime": t1 - t0,
                        "memory_max": mem_max,
                        "oom": False,
                    }
                )
            except (rt.capi.onnxruntime_pybind11_state.RuntimeException, torch.cuda.OutOfMemoryError) as e:
                is_oom = "Out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError)
                results[cfg]["runs"].append(
                    {
                        "event_idx": i,
                        "event_indices": event_indices,
                        "batch_size": len(lengths),
                        "lengths": lengths,
                        "padded_length": padded_length,
                        "size": num_elements,
                        "runtime": None,
                        "memory_max": None,
                        "oom": is_oom,
                    }
                )
                print(f"Skipping event {i} for {cfg} due to error: {e}")
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                continue

            j = particles_to_jets(pred, m_cpu)
            results[cfg]["jets_pt"].append(awkward.to_numpy(awkward.flatten(j.pt)))

            if cfg != baseline:
                if i in baseline_predictions:
                    pred_baseline, m_cpu_baseline = baseline_predictions[i]
                    event_total_err = 0.0
                    event_num_elems = 0
                    for name, idx in [("id", 1), ("momentum", 2), ("pu", 3)]:
                        diff_orig = (pred_baseline[idx][m_cpu_baseline] - pred[idx][m_cpu]).abs().flatten().cpu().float().numpy()
                        is_invalid = ~np.isfinite(diff_orig)
                        results[cfg]["num_invalid"] += np.sum(is_invalid)

                        diff = np.nan_to_num(diff_orig, nan=1e6, posinf=1e6, neginf=1e6)
                        results[cfg][name].append(diff)
                        results[cfg]["total_err"] += np.sum(diff)
                        results[cfg]["num_elems"] += len(diff)
                        event_total_err += np.sum(diff)
                        event_num_elems += len(diff)
                    results[cfg]["runs"][-1]["mae"] = float(event_total_err / event_num_elems)
                else:
                    results[cfg]["runs"][-1]["mae"] = None
            else:
                results[cfg]["runs"][-1]["mae"] = 0.0

        if args.device == "cuda":
            torch.cuda.empty_cache()
        # Clean up model/session to free memory immediately
        if model is not None:
            del model
        if sess is not None:
            if args.profile or args.require_full_gpu:
                provider_profiles[cfg] = finish_profile(sess, args.ort_provider, require_full_gpu=args.require_full_gpu)
            del sess
        gc.collect()
        print(f"Finished validation for {cfg}.")

    # Plotting
    print("Generating comparison plots...")
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(configs))]

    # Runtime vs Event Size Scatter Plot
    plt.figure(figsize=(12, 8))
    for idx, cfg in enumerate(configs):
        plt.scatter(results[cfg]["event_size"], np.array(results[cfg]["runtime"]) * 1000.0, label=cfg, color=colors[idx], alpha=0.5)
    plt.xlabel("Batch size (total valid elements)")
    plt.ylabel("Inference time [ms]")
    plt.legend()
    plt.title("Inference Runtime vs. Batch Elements", y=1.05)
    plt.savefig(os.path.join(args.outdir, "runtime_vs_size.pdf"), bbox_inches="tight")
    plt.close()

    for name in ["id", "momentum", "pu"]:
        # Absolute differences
        plt.figure(figsize=(12, 8))
        for idx, cfg in enumerate(configs):
            if cfg == baseline:
                continue
            d = np.concatenate(results[cfg][name])
            plt.hist(d, bins=np.logspace(-6, 0, 100), label=cfg, histtype="step", lw=2, color=colors[idx])
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(f"Absolute difference in {name} (Ref: {baseline})")
        plt.ylabel("Number of elements")
        plt.legend()
        plt.title(f"Comparison of Absolute Differences: {name}", y=1.05)
        plt.savefig(os.path.join(args.outdir, f"comp_abs_{name}.pdf"), bbox_inches="tight")
        plt.close()

    sorted_runtime_configs = sorted(configs, key=lambda x: np.mean(results[x]["runtime"]))
    avg_runtimes = [np.mean(results[cfg]["runtime"]) * 1000.0 for cfg in sorted_runtime_configs]
    std_runtimes = [np.std(results[cfg]["runtime"]) * 1000.0 for cfg in sorted_runtime_configs]

    # Runtime Distribution
    plt.figure(figsize=(12, 8))
    for idx, cfg in enumerate(configs):
        plt.hist(
            np.array(results[cfg]["runtime"]) * 1000.0,
            bins=np.linspace(0, 2 * np.max(avg_runtimes), 100),
            label=cfg,
            histtype="step",
            lw=2,
            color=colors[idx],
        )
    plt.xlabel("Inference time [ms]")
    plt.ylabel("Number of batches")
    plt.legend()
    plt.title("Distribution of Inference Runtime", y=1.05)
    plt.savefig(os.path.join(args.outdir, "runtime_dist.pdf"), bbox_inches="tight")
    plt.close()

    # Runtime Ranking Bar Chart
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_runtime_configs, avg_runtimes, yerr=std_runtimes, capsize=5, color=[colors[configs.index(cfg)] for cfg in sorted_runtime_configs])
    plt.ylabel("Mean Inference Time [ms]")
    plt.title("Mean Inference Runtime Comparison (Sorted)", y=1.05)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 2.0 * np.max(avg_runtimes))
    plt.savefig(os.path.join(args.outdir, "runtime_ranking.pdf"), bbox_inches="tight")
    plt.close()

    # Final Ranking Bar Chart
    plt.figure(figsize=(12, 8))
    # Calculate Mean Absolute Error (MAE)
    for cfg in configs:
        if results[cfg]["num_elems"] > 0:
            results[cfg]["mae"] = results[cfg]["total_err"] / results[cfg]["num_elems"]
        else:
            results[cfg]["mae"] = 0.0

    sorted_configs = sorted([c for c in configs if c != baseline], key=lambda x: results[x]["mae"])
    maes = [results[cfg]["mae"] for cfg in sorted_configs]

    print(f"\nMean Absolute Error Ranking (relative to Baseline {baseline}):")
    for cfg in sorted_configs:
        invalid_str = f" (Invalid: {results[cfg]['num_invalid']})" if results[cfg]["num_invalid"] > 0 else ""
        print(f"{cfg:20s}: {results[cfg]['mae']:.6e}{invalid_str}")
        # Print per-event MAE for the first few events to debug
        event_maes = [run.get("mae") for run in results[cfg]["runs"][:10]]
        print(f"  First 10 batches MAE: {['{:.2e}'.format(m) if m is not None else 'None' for m in event_maes]}")

    print("\nMean Runtime Summary:")
    for cfg in configs:
        avg_rt = np.mean(results[cfg]["runtime"]) * 1000.0
        std_rt = np.std(results[cfg]["runtime"]) * 1000.0
        print(f"{cfg:20s}: {avg_rt:.2f} +/- {std_rt:.2f} ms")

    print("\nMean Memory Summary:")
    for cfg in configs:
        if results[cfg]["memory_max"]:
            avg_mem = np.mean(results[cfg]["memory_max"])
            std_mem = np.std(results[cfg]["memory_max"])
            print(f"{cfg:20s}: {avg_mem:.2f} +/- {std_mem:.2f} MiB")

    plt.figure(figsize=(12, 8))
    avg_mems = [np.mean(results[cfg]["memory_max"]) if results[cfg]["memory_max"] else 0 for cfg in configs]
    plt.bar(configs, avg_mems, color=[colors[i % len(colors)] for i in range(len(configs))])
    plt.ylabel("Mean Peak Memory [MiB]")
    plt.title(f"Mean Peak Memory Usage ({'GPU' if args.device == 'cuda' else 'CPU'})", y=1.05)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "memory_usage.pdf"), bbox_inches="tight")
    plt.close()

    # Runtime vs. maximum memory scatter plot
    plt.figure(figsize=(12, 8))
    for idx, cfg in enumerate(configs):
        runtimes = np.array(results[cfg]["runtime"]) * 1000.0
        mems = np.array(results[cfg]["memory_max"])
        if len(runtimes) > 0 and len(mems) > 0:
            plt.scatter(mems, runtimes, label=cfg, color=colors[idx], alpha=0.6)
    plt.xlabel("Max Peak Memory [MiB]")
    plt.ylabel("Inference time [ms]")
    plt.title(f"Inference Runtime vs. Max Peak Memory ({'GPU' if args.device == 'cuda' else 'CPU'})", y=1.05)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "runtime_vs_memory.pdf"), bbox_inches="tight")
    plt.close()

    # Loss vs. maximum memory scatter plot. The training loss is model-level,
    # so each point represents a validation scenario for this checkpoint.
    loss_for_plot = train_loss
    if isinstance(loss_for_plot, list) and len(loss_for_plot) > 0:
        loss_for_plot = loss_for_plot[-1]
    if loss_for_plot is not None:
        plt.figure(figsize=(12, 8))
        for idx, cfg in enumerate(configs):
            if results[cfg]["memory_max"]:
                plt.scatter(max(results[cfg]["memory_max"]), loss_for_plot, label=cfg, color=colors[idx], s=80)
        plt.xlabel("Max Peak Memory [MiB]")
        plt.ylabel("Train Loss")
        plt.title(f"Train Loss vs. Max Peak Memory ({'GPU' if args.device == 'cuda' else 'CPU'})", y=1.05)
        plt.xscale("log")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "loss_vs_memory.pdf"), bbox_inches="tight")
        plt.close()

    plt.bar(sorted_configs, maes, color=[colors[configs.index(cfg)] for cfg in sorted_configs])
    plt.ylim(0, 1.5 * np.max(maes) if len(maes) > 0 and np.max(maes) > 0 else 1)
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Model Ranking by MAE (Relative to Baseline {baseline})", y=1.05)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_error_ranking.pdf"), bbox_inches="tight")
    plt.close()

    # Jet pT Distribution Plot
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, figsize=(10, 10))
    b = np.logspace(0, 2, 101)

    hists = {}
    for cfg in configs:
        data = np.concatenate(results[cfg]["jets_pt"]) if results[cfg]["jets_pt"] else np.array([])
        hists[cfg] = to_bh(data, bins=b)

    for idx, cfg in enumerate(configs):
        mplhep.histplot(hists[cfg], label=cfg, lw=1.5, yerr=0, color=colors[idx], ax=a0)

    a0.set_yscale("log")
    a0.set_xscale("log")
    a0.set_ylabel("Number of jets")
    a0.legend(fontsize=8)

    h_ref = hists[baseline]
    for idx, cfg in enumerate(configs):
        ratio = hists[cfg] / h_ref
        mplhep.histplot(ratio, lw=1.5, yerr=0, color=colors[idx], ax=a1)

    a1.set_ylim(0.5, 1.5)
    a1.set_xlim(0, 100)
    a1.set_ylabel(f"vs. {baseline}")
    a1.set_xlabel("jet $p_T$ [GeV]")
    plt.savefig(os.path.join(args.outdir, "jet_pt_distribution.pdf"), bbox_inches="tight")
    plt.close()

    # Save JSON summary
    summary = {
        "num_processed_events": args.num_events,
        "event_size_distribution": all_event_sizes,
        "batch_size": args.batch_size,
        "pad_bin_size": pad_size,
        "runtime_unit": "seconds_per_batch",
        "ort_provider": args.ort_provider,
        "provider_profiles": provider_profiles,
        "require_full_gpu": args.require_full_gpu,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "system": {
            "device": args.device,
            "num_threads": args.num_threads,
            "cpu": get_cpu_info(),
            "gpu": get_gpu_info(),
            "pytorch_version": torch.__version__,
            "onnxruntime_version": rt.__version__,
        },
        "scenarios": {cfg: results[cfg]["runs"] for cfg in configs},
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Done. Results saved in {args.outdir}")


if __name__ == "__main__":
    main()
