"""CPU-only diagnostic for cms-sw/cmssw#52009; no production model changes.

Run from the repository root with the existing ONNX CPU environment:
  PYTHONPATH=. envs/ort-cpu/.venv/bin/python scripts/reproduce_attention_padding.py \
      --deployed local_test_data/masking_review/deployed.onnx

Inputs and weights for the current-code check are synthetic. The optional
deployed-model check uses synthetic inputs and rewrites only an in-memory copy.
This checks padding invariance, not physics performance or MIGraphX support.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF, PreLnSelfAttentionLayer


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def compare(reference, candidate):
    n = reference[0].shape[1]
    candidate = [x[:, :n] for x in candidate]
    return {
        "max_abs_diff": {k: float(np.max(np.abs(a - b))) for k, a, b in zip(("bid", "id", "momentum", "pu"), reference, candidate)},
        "pid_changes": int(np.sum(reference[1].argmax(-1) != candidate[1].argmax(-1))),
        "finite": all(bool(np.isfinite(x).all()) for x in candidate),
    }


def run_padded(runtime, features, padding):
    return runtime.run(
        None,
        {
            "Xfeat_normed": np.pad(features, ((0, 0), (0, padding), (0, 0))),
            "mask": np.pad(np.ones(features.shape[:2], np.float32), ((0, 0), (0, padding))),
        },
    )


def current_code(outdir):
    torch.manual_seed(7)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {"type": "attention", "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"}},
        }
    )
    model = MLPF(config).eval()
    features = torch.randn(1, 30, config.input_dim)
    features[..., 0] = 1
    features[..., 1] = features[..., 1].exp()
    features[..., 5] = features[..., 5].exp()
    mask = torch.ones(1, 30)
    padded = torch.nn.functional.pad(features, (0, 0, 0, 20))
    padded_mask = torch.nn.functional.pad(mask, (0, 20))

    def predict(x, valid):
        with torch.no_grad():
            return [y.numpy() for y in model(x, valid)]

    baseline = predict(features, mask)
    padded_outputs = predict(padded, padded_mask)
    result = {"pytorch_padding": compare(baseline, padded_outputs)}
    path = outdir / "current_math.onnx"
    torch.onnx.export(
        model,
        (features, mask),
        str(path),
        input_names=["Xfeat_normed", "mask"],
        output_names=["bid", "id", "momentum", "pu"],
        dynamic_axes={"Xfeat_normed": {0: "batch", 1: "elements"}, "mask": {0: "batch", 1: "elements"}},
        opset_version=20,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    result["onnx_ops"] = dict(Counter(n.op_type for n in graph.graph.node))
    runtime = session(graph)
    result["onnx_padding"] = compare(run_padded(runtime, features.numpy(), 0), run_padded(runtime, features.numpy(), 20))
    result["onnx_vs_pytorch_padded"] = compare(padded_outputs, run_padded(runtime, features.numpy(), 20))

    # Isolate the missing caller mask. Hooks are local to this diagnostic model.
    def capture_mask(layer, args):
        layer.mha._diagnostic_mask = args[1]

    def supply_mask(attention, args, kwargs):
        kwargs["key_padding_mask"] = ~attention._diagnostic_mask.bool()
        return args, kwargs

    handles = []
    for layer in model.modules():
        if isinstance(layer, PreLnSelfAttentionLayer):
            handles.append(layer.register_forward_pre_hook(capture_mask))
            handles.append(layer.mha.register_forward_pre_hook(supply_mask, with_kwargs=True))
    try:
        masked_baseline = predict(features, mask)
        result["supplied_key_mask_unpadded"] = compare(baseline, masked_baseline)
        result["supplied_key_mask_padding"] = compare(masked_baseline, predict(padded, padded_mask))
    finally:
        for handle in handles:
            handle.remove()
    return result


def deployed_model(path):
    model = onnx.load(path)
    result = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    features = np.random.default_rng(7).normal(size=(1, 300, 55)).astype(np.float32)
    features[..., 0] = 1
    features[..., 1] = np.exp(features[..., 1])
    features[..., 5] = np.exp(features[..., 5])
    runtime = session(model)
    original = run_padded(runtime, features, 0)
    result["unpadded_finite"] = all(bool(np.isfinite(x).all()) for x in original)
    try:
        run_padded(runtime, features, 1)
        result["one_pad_error"] = None
    except ort.capi.onnxruntime_pybind11_state.InvalidArgument as error:
        result["one_pad_error"] = str(error)

    # Bypass the particular broken assignment in the 20250722 model.
    scatter = next(n for n in model.graph.node if n.name == "/ScatterND")
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == scatter.output[0]:
                node.input[i] = scatter.input[0]
    while True:
        used = {x for n in model.graph.node for x in n.input} | {x.name for x in model.graph.output}
        dead = [n for n in model.graph.node if not any(x in used for x in n.output)]
        if not dead:
            break
        for node in dead:
            model.graph.node.remove(node)
    runtime = session(model)
    baseline = run_padded(runtime, features, 0)
    result["bypass_unpadded"] = compare(original, baseline)
    result["bypass_padding"] = {str(p): compare(baseline, run_padded(runtime, features, p)) for p in (1, 50, 700, 2000)}

    # Keep the existing fp16 MHA and supply its int32 valid-key mask (input 5).
    # This is a CPU diagnostic, not the issue's fp32 MIGraphX graph rewrite.
    for node in model.graph.node:
        if node.op_type == "SDPA":
            node.input.append("mask")
    for function in model.functions:
        if function.name == "SDPA":
            function.input.append("mask")
            function.node.insert(0, onnx.helper.make_node("Cast", ["mask"], ["valid_keys"], to=onnx.TensorProto.INT32))
            for node in function.node:
                if node.op_type == "MultiHeadAttention":
                    node.input.extend(["", "valid_keys"])
    onnx.checker.check_model(model)
    runtime = session(model)
    masked_baseline = run_padded(runtime, features, 0)
    result["key_mask_unpadded"] = compare(original, masked_baseline)
    result["key_mask_padding"] = {str(p): compare(masked_baseline, run_padded(runtime, features, p)) for p in (1, 50, 700, 2000)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployed", type=Path, help="Optional original CMSSW 20250722 ONNX model")
    parser.add_argument("--outdir", type=Path, default=Path("local_test_data/masking_review"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    result = {
        "environment": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider"},
        "current": current_code(args.outdir),
    }
    if args.deployed:
        result["deployed"] = deployed_model(args.deployed)
    report = json.dumps(result, indent=2)
    (args.outdir / "results.json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
