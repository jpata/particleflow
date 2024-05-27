import torch
import time
import onnxruntime
import pathlib
import onnxscript
import onnx
import math
import numpy


dtype_map = {
    numpy.dtype("float32"): onnx.TensorProto.FLOAT,
    numpy.dtype("bool"): onnx.TensorProto.BOOL,
}


class Model(torch.nn.Module):
    def forward(self, query_states, key_states, value_states, mask):
        query_states = query_states
        key_states = key_states
        value_states = value_states
        return torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
        )


model = Model()
model.eval()

# (B, num_heads, N, head_dim)
query_states = torch.randn(1, 32, 256, 64)
key_states = torch.randn(1, 32, 256, 64)
value_states = torch.randn(1, 32, 256, 64)
mask = torch.randn(1, 32, 256, 1)

torch_out = model(query_states, key_states, value_states, mask)
print(torch_out.shape)
print(torch_out)

# Another reference perf comparison.
# torch.onnx.export(
#     model,
#     (query_states, key_states, value_states, mask),
#     "sdpa.onnx",
#     verbose=True,
#     opset_version=14,
#     input_names=["query_states", "key_states", "value_states", "mask"],
# )

model_dir = "multihead_attention"
fused_model_name = "multihead_attention.onnx"
fused_model_path = f"{model_dir}/{fused_model_name}"

unfused_model_name = "unfused_multihead_attention.onnx"
unfused_model_path = f"{model_dir}/{unfused_model_name}"

pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

msft_op = onnxscript.values.Opset("com.microsoft", 1)
op = onnxscript.opset13

sqrt_head_dim = math.sqrt(query_states.shape[-1])

query_states_ort = query_states.numpy()
key_states_ort = key_states.numpy()
value_states_ort = value_states.numpy()
attention_mask_ort = mask.numpy()

ort_inputs = {
    "query_states": query_states_ort,
    "key_states": key_states_ort,
    "value_states": value_states_ort,
    "mask": attention_mask_ort,
}

print("Benchmarking PT sdpa and ORT MultiHeadAttention...")


def run_pt():
    # warmup
    for _ in range(30):
        model(query_states, key_states, value_states, mask)

    total_time = 0
    for _ in range(1000):
        start_time = time.perf_counter()
        model(query_states, key_states, value_states, mask)
        total_time += time.perf_counter() - start_time
    return total_time


total_time = run_pt()

print("PT eager:")
print(f"Total time: {total_time:.2f}s")


def mha_onnx_model(query_states, key_states, value_states, mask):
    # query_states = op.Reshape(query_states, shape=[1, 32, 128, 1, 64])
    # key_states = op.Reshape(key_states, shape=[1, 32, 128, 1, 64])
    # value_states = op.Reshape(value_states, shape=[1, 32, 128, 1, 64])
    # qkv = op.Concat(query_states, key_states, value_states, axis=3)

    # query_states = op.Reshape(op.Transpose(query_states, perm=[1,0,2]), shape=[32,1024,64])
    # key_states = op.Reshape(op.Transpose(key_states, perm=[1,0,2]), shape=[32,1024,64])
    # value_states = op.Reshape(op.Transpose(value_states, perm=[1,0,2]), shape=[32,1024,64])

    # 1, 32, 256, 64 -> 1, 256, 32, 64 -> 1,256,2048
    query_states = op.Reshape(op.Transpose(query_states, perm=[0, 2, 1, 3]), shape=[1, 256, 2048])
    key_states = op.Reshape(op.Transpose(key_states, perm=[0, 2, 1, 3]), shape=[1, 256, 2048])
    value_states = op.Reshape(op.Transpose(value_states, perm=[0, 2, 1, 3]), shape=[1, 256, 2048])
    output, _, _ = msft_op.MultiHeadAttention(
        query_states,
        key_states,
        value_states,
        num_heads=32,
    )
    output = op.Reshape(output, shape=[1, 256, 32, 64])
    output = op.Transpose(output, perm=[0, 2, 1, 3])
    return output


def unfused_onnx_model(query_states, key_states, value_states, mask):

    scale = op.Constant(value_float=sqrt_head_dim)

    attn_weights = op.MatMul(query_states, op.Transpose(key_states, perm=[0, 1, 3, 2])) / scale
    # attn_weights = op.Add(attn_weights, mask)
    attn_weights = op.Softmax(attn_weights, axis=-1)
    attn_output = op.MatMul(attn_weights, value_states)
    return attn_output


def serialize_model(model_func, model_name, ort_inputs):
    model_path = f"{model_dir}/{model_name}"
    model_proto = onnxscript.script(onnxscript.opset13, default_opset=onnxscript.opset13)(model_func).to_model_proto()

    for i, value in enumerate(ort_inputs.values()):
        model_proto.graph.input[i].type.CopyFrom(
            onnx.helper.make_tensor_type_proto(
                shape=value.shape,
                elem_type=dtype_map[value.dtype],
            )
        )
    model_proto.graph.output[0].type.CopyFrom(
        onnx.helper.make_tensor_type_proto(
            shape=[32, 1024, 64],
            elem_type=onnx.TensorProto.FLOAT,
        )
    )

    onnx.save(model_proto, model_path)
    return model_proto, model_path


def save_tensor_data(numpy_tensor, output_path):
    from onnx import numpy_helper

    proto_tensor = numpy_helper.from_array(numpy_tensor)
    with open(output_path, "wb") as f:
        f.write(proto_tensor.SerializeToString())


def serialize_inputs_outputs(model_dir, onnx_inputs, onnx_outputs):
    test_data_dir = pathlib.Path(f"{model_dir}/test_data_set_0")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    for i, onnx_input in enumerate(onnx_inputs.values()):
        save_tensor_data(onnx_input, str(test_data_dir / f"input_{i}.pb"))

    for i, onnx_output in enumerate(onnx_outputs):
        save_tensor_data(onnx_output, str(test_data_dir / f"output_{i}.pb"))


def run_ort(model_func, model_name, ort_inputs):
    # Serialize model
    model_proto, model_path = serialize_model(model_func, model_name, ort_inputs)

    # Serialize inputs and outputs
    sess = onnxruntime.InferenceSession(model_path)
    ort_outputs = sess.run(None, ort_inputs)

    # Parity
    torch.testing.assert_close(torch_out, torch.tensor(ort_outputs[0]), rtol=1e-3, atol=1e-3)

    serialize_inputs_outputs(model_dir, ort_inputs, ort_outputs)

    # warmup
    for _ in range(30):
        sess.run(None, ort_inputs)

    total_time = 0
    for _ in range(10):
        start_time = time.perf_counter()
        sess.run(None, ort_inputs)
        total_time += time.perf_counter() - start_time

    print(f"ORT {model_name}")
    print(f"Total time: {total_time:.2f}s")


run_ort(unfused_onnx_model, unfused_model_name, ort_inputs)
run_ort(mha_onnx_model, fused_model_name, ort_inputs)
