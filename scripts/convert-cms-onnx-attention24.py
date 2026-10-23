"""Upgrade the July 2025 CMS MLPF artifact, retaining CUDA FP16 FlashAttention.

Only the pinned source artifact is supported. Inputs must use binary, right-padded
masks. Learned weights and the original CUDA attention precision are preserved.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

SOURCE_SHA256 = "bff46e3bc918bff466686e41f81530364f80a28af5a98e041d1bf726ea5c071a"


def convert(source: Path, destination: Path):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unsupported source artifact: SHA256 does not match the July 2025 CMS model")
    model = onnx.load(source)
    weights = {x.name: x.SerializeToString() for x in model.graph.initializer}
    model = onnx.version_converter.convert_version(model, 24)

    def constant(name, value):
        return h.make_node("Constant", [], [name], value=nh.from_array(np.asarray(value)))

    nodes = [
        h.make_node("Cast", ["mask"], ["patch_mask_i64"], to=T.INT64),
        constant("patch_axis", np.array([1], np.int64)),
        h.make_node("ReduceSum", ["patch_mask_i64", "patch_axis"], ["patch_lengths"], keepdims=0),
    ]
    count = 0
    for node in model.graph.node:
        if node.name == "/ScatterND":
            # The original export captured an empty update tensor. Where also
            # restores zero log-features for padded elements without indexing.
            nodes += [
                constant("patch_zero", np.float32(0)),
                h.make_node(
                    "Where",
                    ["/conv_id.0/Unsqueeze_output_0", node.input[0], "patch_zero"],
                    list(node.output),
                    name="patch_padding_where",
                ),
            ]
        elif node.op_type == "SDPA":
            count += 1
            prefix = node.name + "_patch"
            nodes += [
                constant(prefix + "_shape4", np.array([0, 0, 16, 16], np.int64)),
                constant(prefix + "_shape3", np.array([0, 0, 256], np.int64)),
            ]
            inputs = []
            for i, value in enumerate(node.input):
                name = prefix + f"_{i}"
                nodes += [
                    h.make_node("Cast", [value], [name + "_half"], to=T.FLOAT16),
                    h.make_node(
                        "Reshape",
                        [name + "_half", prefix + "_shape4"],
                        [name + "_reshape"],
                    ),
                    h.make_node(
                        "Transpose",
                        [name + "_reshape"],
                        [name + "_qkv"],
                        perm=[0, 2, 1, 3],
                    ),
                ]
                inputs.append(name + "_qkv")
            # Input 6 is nonpad_kv_seqlen; a dense mask would prevent CUDA FlashAttention.
            nodes += [
                h.make_node(
                    "Attention",
                    inputs + ["", "", "", "patch_lengths"],
                    [prefix + "_out"],
                    is_causal=0,
                    softmax_precision=T.FLOAT,
                    name=prefix + "_attention",
                ),
                h.make_node(
                    "Transpose",
                    [prefix + "_out"],
                    [prefix + "_transposed"],
                    perm=[0, 2, 1, 3],
                ),
                h.make_node(
                    "Reshape",
                    [prefix + "_transposed", prefix + "_shape3"],
                    [prefix + "_dense"],
                ),
                h.make_node("Cast", [prefix + "_dense"], list(node.output), to=T.FLOAT),
            ]
        else:
            nodes.append(node)
    if count != 6:
        raise ValueError(f"Expected six attention blocks, found {count}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.functions[:]
    del model.opset_import[:]
    model.opset_import.extend([h.make_opsetid("", 24)])
    # Remove the now-unused ScatterND index/update chain.
    while True:
        used = {x for n in model.graph.node for x in n.input} | {x.name for x in model.graph.output}
        dead = [n for n in model.graph.node if not any(o in used for o in n.output)]
        if not dead:
            break
        for node in dead:
            model.graph.node.remove(node)
    assert weights == {x.name: x.SerializeToString() for x in model.graph.initializer}
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata.update(
        {
            "source_sha256": SOURCE_SHA256,
            "attention_upgrade": "Attention-24; FP16 QKV, FP32 softmax; unchanged learned weights",
            "mask_contract": "Binary mask [B,L], valid prefix followed by right padding; discard padded outputs",
        }
    )
    h.set_model_props(model, metadata)
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    print(f"{hashlib.sha256(destination.read_bytes()).hexdigest()}  {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
