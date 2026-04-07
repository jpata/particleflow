import sys

# Support for GPU executor: set onnxruntime path before imports
if "--device" in sys.argv:
    idx = sys.argv.index("--device")
    if sys.argv[idx + 1] == "cuda":
        print("device=cuda, using onnxruntime-gpu from /opt/onnxruntime-gpu/lib/python3.12/site-packages/")
        sys.path.insert(0, "/opt/onnxruntime-gpu/lib/python3.12/site-packages/")

import os
import time
import gc
import json
import platform
import subprocess

# Ensure mlpf is in the path
sys.path.append(os.getcwd())

import copy
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
import onnx
import onnxruntime as rt

print("onnxruntime", rt.__path__)
import onnxscript
from onnxscript import opset20 as op
import awkward
import vector
import fastjet
from mlpf.model.mlpf import MLPF
from mlpf.conf import MLPFConfig, ModelType, AttentionType
from mlpf.model.utils import unpack_predictions
from mlpf.plotting.plot_utils import ELEM_NAMES_CMS, CLASS_NAMES_CMS, CLASS_NAMES_CLIC

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
        if k in ["use_simplified_attention", "export_onnx_fused", "save_attention", "attention_type"]:
            if config.model.type == ModelType.ATTENTION:
                # If the original model was linear, we should keep it linear
                if k == "attention_type" and config.model.attention.attention_type == AttentionType.LINEAR:
                    continue
                setattr(config.model.attention, k, v)
        elif hasattr(config, k):
            setattr(config, k, v)
        elif hasattr(config.model, k):
            setattr(config.model, k, v)
    return config


SUPPORTED_CONFIGS = [
    "PT_MATH_FP32",
    "PT_MATH_FP16",
    "PT_FLASH_FP16",
    "ONNX_MATH_FP32",
    "ONNX_MATH_FP16",
    "ONNX_FLASH_FP32_FP16",
    "ONNX_FLASH_FP16",
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
    parser.add_argument("--pad-bin-size", type=int, default=0, help="Pad events to the nearest multiple of this size")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=SUPPORTED_CONFIGS,
        default=SUPPORTED_CONFIGS,
        help="List of configurations to run",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Number of CPU threads to use")
    return parser.parse_args()


def particles_to_jets(pred, mask):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    ypred = unpack_predictions(pred)
    for k, v in ypred.items():
        ypred[k] = v[mask].detach().cpu().contiguous().numpy()

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
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.outdir, exist_ok=True)
    mplhep.style.use("CMS")

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

    NUM_HEADS = model_kwargs.model.attention.num_heads if model_kwargs.model.type == ModelType.ATTENTION else 1
    input_dim = model_kwargs.input_dim

    opset_version = 20

    configs = args.configs
    baseline = "PT_MATH_FP32" if "PT_MATH_FP32" in configs else configs[0]
    # Reorder configs to run baseline first for comparison
    configs = [baseline] + [c for c in configs if c != baseline]

    # Model Variants for Export (always in float32 for export reliability)
    model_kwargs_export = model_kwargs.model_copy(deep=True)
    if model_kwargs_export.model.type == ModelType.ATTENTION:
        if model_kwargs_export.model.attention.attention_type != AttentionType.LINEAR:
            model_kwargs_export.model.attention.attention_type = AttentionType.MATH

    # Math attention model (for math exports)
    if any(cfg in configs for cfg in ["ONNX_MATH_FP32", "ONNX_MATH_FP16"]):
        print("Initializing Math attention model for export...")
        model_math = MLPF(
            config=make_mlpf_config(model_kwargs_export, use_simplified_attention=True, export_onnx_fused=False),
        )
        model_math.eval()
        model_math.load_state_dict(model_state["model_state_dict"], strict=False)
        model_math = model_math.to(device=args.device)

    # Fused attention model (for fused export)
    if any(cfg in configs for cfg in ["ONNX_FLASH_FP32_FP16", "ONNX_FLASH_FP16"]):
        print("Initializing Fused attention model for export...")
        model_fused = MLPF(
            config=make_mlpf_config(model_kwargs_export, use_simplified_attention=True, export_onnx_fused=True),
        )
        model_fused.eval()
        model_fused.load_state_dict(model_state["model_state_dict"], strict=False)
        model_fused = model_fused.to(device=args.device)

    # Dummy inputs
    dummy_features = torch.randn(1, 256, input_dim).float().to(args.device)
    dummy_mask = (torch.randn(1, 256) > 0.5).float().to(args.device)

    # 1. Export ONNX Math FP32
    if "ONNX_MATH_FP32" in configs:
        path_math_fp32 = os.path.join(args.outdir, "model_math_fp32.onnx")
        print(f"Exporting ONNX Math FP32 to {path_math_fp32}...")
        torch.onnx.export(
            model_math,
            (dummy_features, dummy_mask),
            path_math_fp32,
            opset_version=opset_version,
            verbose=False,
            input_names=["Xfeat_normed", "mask"],
            output_names=["bid", "id", "momentum", "pu"],
            dynamic_axes={"Xfeat_normed": {0: "num_batch", 1: "num_elements"}, "mask": {0: "num_batch", 1: "num_elements"}},
            dynamo=False,
        )

    # 2. Export ONNX Math FP16
    if "ONNX_MATH_FP16" in configs:
        path_math_fp16 = os.path.join(args.outdir, "model_math_fp16.onnx")
        print(f"Exporting ONNX Math FP16 to {path_math_fp16}...")
        # Cast to half for export
        model_math_half = copy.deepcopy(model_math).half()
        torch.onnx.export(
            model_math_half,
            (dummy_features.half(), dummy_mask.half()),
            path_math_fp16,
            opset_version=opset_version,
            verbose=False,
            input_names=["Xfeat_normed", "mask"],
            output_names=["bid", "id", "momentum", "pu"],
            dynamic_axes={"Xfeat_normed": {0: "num_batch", 1: "num_elements"}, "mask": {0: "num_batch", 1: "num_elements"}},
            dynamo=False,
        )
        del model_math_half

    # 3. Export ONNX Fused Flash Mixed/FP16
    if any(cfg in configs for cfg in ["ONNX_FLASH_FP32_FP16", "ONNX_FLASH_FP16"]):
        print("Registering custom SDPA op for ONNX export...")
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        @onnxscript.script(custom_opset)
        def SDPA(query, key, value):
            q16 = op.Cast(query, to=onnx.TensorProto.FLOAT16)
            k16 = op.Cast(key, to=onnx.TensorProto.FLOAT16)
            v16 = op.Cast(value, to=onnx.TensorProto.FLOAT16)
            output, _, _ = msft_op.MultiHeadAttention(q16, k16, v16, num_heads=NUM_HEADS)
            return op.CastLike(output, query)

        def custom_scaled_dot_product_attention(g, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            return g.onnxscript_op(SDPA, query, key, value).setType(query.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::scaled_dot_product_attention",
            symbolic_fn=custom_scaled_dot_product_attention,
            opset_version=opset_version,
        )

        # 3a. Export ONNX Fused Flash Mixed (FP32 model, FP16 attention)
        if "ONNX_FLASH_FP32_FP16" in configs:
            path_fused_fp32_fp16 = os.path.join(args.outdir, "model_fused_fp32_fp16.onnx")
            print(f"Exporting ONNX Fused Flash Mixed (FP32/FP16) to {path_fused_fp32_fp16}...")
            torch.onnx.export(
                model_fused,
                (dummy_features, dummy_mask),
                path_fused_fp32_fp16,
                opset_version=opset_version,
                verbose=False,
                input_names=["Xfeat_normed", "mask"],
                output_names=["bid", "id", "momentum", "pu"],
                dynamic_axes={"Xfeat_normed": {0: "num_batch", 1: "num_elements"}, "mask": {0: "num_batch", 1: "num_elements"}},
                dynamo=False,
            )

        # 3b. Export ONNX Fused Flash FP16 (Full FP16)
        if "ONNX_FLASH_FP16" in configs:
            path_fused_fp16 = os.path.join(args.outdir, "model_fused_fp16.onnx")
            print(f"Exporting ONNX Fused Flash FP16 to {path_fused_fp16}...")
            model_fused_half = copy.deepcopy(model_fused).half()
            torch.onnx.export(
                model_fused_half,
                (dummy_features.half(), dummy_mask.half()),
                path_fused_fp16,
                opset_version=opset_version,
                verbose=False,
                input_names=["Xfeat_normed", "mask"],
                output_names=["bid", "id", "momentum", "pu"],
                dynamic_axes={"Xfeat_normed": {0: "num_batch", 1: "num_elements"}, "mask": {0: "num_batch", 1: "num_elements"}},
                dynamo=False,
            )
            del model_fused_half

    # Initialize ONNX sessions
    print("Initializing ONNX sessions...")
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = args.num_threads
    sess_options.inter_op_num_threads = args.num_threads
    execution_provider = "CPUExecutionProvider" if args.device == "cpu" else "CUDAExecutionProvider"

    if "ONNX_MATH_FP32" in configs:
        print(f"Creating ONNX session for Math FP32 using {execution_provider}...")
        sess_math_fp32 = rt.InferenceSession(path_math_fp32, sess_options, providers=[execution_provider])
    if "ONNX_MATH_FP16" in configs:
        print(f"Creating ONNX session for Math FP16 using {execution_provider}...")
        sess_math_fp16 = rt.InferenceSession(path_math_fp16, sess_options, providers=[execution_provider])
    if "ONNX_FLASH_FP32_FP16" in configs:
        print(f"Creating ONNX session for Fused Mixed using {execution_provider}...")
        sess_fused_fp32_fp16 = rt.InferenceSession(path_fused_fp32_fp16, sess_options, providers=[execution_provider])
    if "ONNX_FLASH_FP16" in configs:
        print(f"Creating ONNX session for Fused FP16 using {execution_provider}...")
        sess_fused_fp16 = rt.InferenceSession(path_fused_fp16, sess_options, providers=[execution_provider])

    # PyTorch Math Model
    if any(cfg in configs for cfg in ["PT_MATH_FP32", "PT_MATH_FP16"]):
        print("Initializing PyTorch Math model...")
        model_kwargs_math = model_kwargs.model_copy(deep=True)

        model_pt_math = MLPF(
            config=make_mlpf_config(
                model_kwargs_math, attention_type="math", use_simplified_attention=False, export_onnx_fused=False, save_attention=False
            ),
        )
        model_pt_math.eval()
        model_pt_math.load_state_dict(model_state["model_state_dict"], strict=False)
        model_pt_math = model_pt_math.to(device=args.device)

    # PyTorch Flash Model
    if "PT_FLASH_FP16" in configs:
        print("Initializing PyTorch Flash model...")
        model_kwargs_flash = model_kwargs.model_copy(deep=True)

        model_pt_flash = MLPF(
            config=make_mlpf_config(
                model_kwargs_flash,
                attention_type="flash" if args.device == "cuda" else "math",
                use_simplified_attention=False,
                export_onnx_fused=False,
                save_attention=False,
            ),
        )
        model_pt_flash.eval()
        model_pt_flash.load_state_dict(model_state["model_state_dict"], strict=False)
        model_pt_flash = model_pt_flash.to(device=args.device)

    # Validation
    print(f"Loading dataset {args.dataset} from {args.data_dir}...")
    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    ds = builder.as_data_source(split="train")

    # Warmup
    print("Performing warmup...")
    elem_warmup = ds[0]
    X_warmup = torch.tensor(elem_warmup["X"]).to(torch.float32).to(args.device)
    if args.pad_bin_size > 0:
        if X_warmup.shape[0] % args.pad_bin_size != 0:
            num_to_pad = args.pad_bin_size - (X_warmup.shape[0] % args.pad_bin_size)
            X_warmup = torch.nn.functional.pad(X_warmup, (0, 0, 0, num_to_pad), mode="constant", value=0)

    X_warmup = X_warmup.unsqueeze(0).contiguous()
    mask_warmup = X_warmup[:, :, 0] != 0
    mask_f_warmup = mask_warmup.float().cpu().numpy()
    X_warmup_np = X_warmup.cpu().numpy()
    X_warmup_np_fp16 = X_warmup_np.astype(np.float16)
    mask_f_warmup_fp16 = mask_f_warmup.astype(np.float16)

    for _ in tqdm(range(10)):
        with torch.no_grad():
            if any(cfg in configs for cfg in ["PT_MATH_FP32", "PT_MATH_FP16"]):
                _ = model_pt_math(X_warmup, mask_warmup)
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                    _ = model_pt_math(X_warmup, mask_warmup)
            if "PT_FLASH_FP16" in configs:
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                    _ = model_pt_flash(X_warmup, mask_warmup)
        if "ONNX_MATH_FP32" in configs:
            _ = sess_math_fp32.run(None, {"Xfeat_normed": X_warmup_np, "mask": mask_f_warmup})
        if "ONNX_MATH_FP16" in configs:
            _ = sess_math_fp16.run(None, {"Xfeat_normed": X_warmup_np_fp16, "mask": mask_f_warmup_fp16})
        if "ONNX_FLASH_FP32_FP16" in configs:
            _ = sess_fused_fp32_fp16.run(None, {"Xfeat_normed": X_warmup_np, "mask": mask_f_warmup})
        if "ONNX_FLASH_FP16" in configs:
            _ = sess_fused_fp16.run(None, {"Xfeat_normed": X_warmup_np_fp16, "mask": mask_f_warmup_fp16})
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete. Starting validation scenarios.")

    results = {
        cfg: {
            "id": [],
            "momentum": [],
            "pu": [],
            "total_err": 0.0,
            "num_elems": 0,
            "num_invalid": 0,
            "runtime": [],
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

    for cfg in configs:
        print(f"Running validation for {cfg}...")
        for i in tqdm(range(args.num_events)):
            elem = ds[i]
            X_features = torch.tensor(elem["X"]).to(torch.float32).to(args.device)
            num_elements = X_features.shape[0]

            # Pad elements to the nearest multiple if specified
            if args.pad_bin_size > 0:
                if num_elements % args.pad_bin_size != 0:
                    num_to_pad = args.pad_bin_size - (num_elements % args.pad_bin_size)
                    X_features = torch.nn.functional.pad(X_features, (0, 0, 0, num_to_pad), mode="constant", value=0)

            # Clip features to avoid FP16 overflow (max value ~65500)
            X_features = torch.clamp(X_features, min=-60000, max=60000)

            X_features_padded = X_features.unsqueeze(0).contiguous()
            mask = X_features_padded[:, :, 0] != 0
            mask_f = mask.float()
            m_cpu = mask.cpu()

            # Pre-convert to numpy for ONNX
            X_features_np = X_features_padded.cpu().numpy()
            mask_f_np = mask_f.cpu().numpy()

            try:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                if cfg == "PT_MATH_FP32":
                    with torch.no_grad():
                        pred = model_pt_math(X_features_padded, mask)
                elif cfg == "PT_MATH_FP16":
                    with torch.no_grad():
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            pred = model_pt_math(X_features_padded, mask)
                elif cfg == "PT_FLASH_FP16":
                    with torch.no_grad():
                        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                            pred = model_pt_flash(X_features_padded, mask)
                elif cfg == "ONNX_MATH_FP32":
                    pred = sess_math_fp32.run(None, {"Xfeat_normed": X_features_np, "mask": mask_f_np})
                    pred = tuple(torch.tensor(p) for p in pred)
                elif cfg == "ONNX_MATH_FP16":
                    X_features_np_fp16 = X_features_np.astype(np.float16)
                    mask_f_np_fp16 = mask_f_np.astype(np.float16)
                    pred = sess_math_fp16.run(None, {"Xfeat_normed": X_features_np_fp16, "mask": mask_f_np_fp16})
                    pred = tuple(torch.tensor(p).float() for p in pred)
                elif cfg == "ONNX_FLASH_FP32_FP16":
                    pred = sess_fused_fp32_fp16.run(None, {"Xfeat_normed": X_features_np, "mask": mask_f_np})
                    pred = tuple(torch.tensor(p).float() for p in pred)
                elif cfg == "ONNX_FLASH_FP16":
                    X_features_np_fp16 = X_features_np.astype(np.float16)
                    mask_f_np_fp16 = mask_f_np.astype(np.float16)
                    pred = sess_fused_fp16.run(None, {"Xfeat_normed": X_features_np_fp16, "mask": mask_f_np_fp16})
                    pred = tuple(torch.tensor(p).float() for p in pred)

                if args.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                if "PT" in cfg:
                    pred = tuple(p.cpu().float() for p in pred)

                if cfg == baseline:
                    baseline_predictions[i] = (pred, m_cpu)

                results[cfg]["runtime"].append(t1 - t0)
                results[cfg]["event_size"].append(num_elements)
                results[cfg]["runs"].append({"event_idx": i, "size": num_elements, "runtime": t1 - t0, "oom": False})
            except (rt.capi.onnxruntime_pybind11_state.RuntimeException, torch.cuda.OutOfMemoryError) as e:
                is_oom = "Out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError)
                results[cfg]["runs"].append({"event_idx": i, "size": num_elements, "runtime": None, "oom": is_oom})
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
                        diff_orig = (pred_baseline[idx][m_cpu_baseline] - pred[idx][m_cpu]).abs().flatten().numpy()
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
        gc.collect()
        print(f"Finished validation for {cfg}.")

    # Plotting
    print("Generating comparison plots...")
    colors = ["black", "blue", "cyan", "red", "orange", "magenta", "green"]

    # Runtime vs Event Size Scatter Plot
    plt.figure(figsize=(12, 8))
    for idx, cfg in enumerate(configs):
        plt.scatter(results[cfg]["event_size"], np.array(results[cfg]["runtime"]) * 1000.0, label=cfg, color=colors[idx], alpha=0.5)
    plt.xlabel("Event size (number of elements)")
    plt.ylabel("Inference time [ms]")
    plt.legend()
    plt.title("Inference Runtime vs. Event Size", y=1.05)
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
    plt.ylabel("Number of events")
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

    print("\nMean Runtime Summary:")
    for cfg in configs:
        avg_rt = np.mean(results[cfg]["runtime"]) * 1000.0
        std_rt = np.std(results[cfg]["runtime"]) * 1000.0
        print(f"{cfg:20s}: {avg_rt:.2f} +/- {std_rt:.2f} ms")

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
