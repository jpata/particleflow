# Export and validate ONNX

Use ONNX validation to check numerical agreement with PyTorch before measuring deployment latency or memory. `scripts/cms-validate-onnx.py` supports CMS, CLD, and CLIC TFDS inputs despite its name. Start with FP32 math attention on CPU, then qualify the target runtime and precision.

## Supported export contract

`mlpf.model.onnx_attention.export_mlpf` exports an evaluation-mode model with elementwise predictions:

| Tensor | Shape and meaning |
|---|---|
| `Xfeat_normed` | Finite features `[B,L,F]` matching the saved model's feature layout |
| `mask` | Binary validity mask `[B,L]`, with at least one valid element per event and right padding |
| `bid`, `id`, `momentum`, `pu` | Four separate prediction heads `[B,L,head_width]` |

Batch size and sequence length are dynamic. Masks must contain a valid prefix followed by padding; holes, left padding, and empty events are invalid. Validation tools reject these inputs. Deployment callers must enforce the same contract because the exported graph cannot raise those errors.

The exporter sanitizes padded features, excludes padded keys from attention, and materializes each output head with exactly zero values on padding. Consumers should still use the mask to select real predictions. Particle-set decoders have a different output contract and are explicitly rejected by this export helper.

## Choose an attention representation

Export representation and eager PyTorch backend are separate choices:

| Representation | Graph | Intended use |
|---|---|---|
| `decomposed` | Opset 20, explicit scaled MatMul, FP32 Softmax, MatMul, and key masking | Portable numerical baseline; AMD attention fusion with a qualified runtime |
| `attention24` | Standard ONNX Attention at opset 24 with `nonpad_kv_seqlen: int64[B]` | Length-aware attention, including CUDA FlashAttention where supported |

For an FP32 model in `eval()` mode and finite example tensors:

```python
import torch
from mlpf.model.onnx_attention import export_mlpf

export_mlpf(model, features, mask, "model_math_fp32.onnx",
            attention_format="decomposed")
export_mlpf(model, features, mask, "model_attention24_mixed.onnx",
            attention_format="attention24", attention_dtype=torch.float16)
```

`attention_dtype` changes the attention computation; other operations and public I/O retain the model/example dtype. Omitted `attention_format` follows `export_onnx_fused`, otherwise defaulting to decomposed attention. Export temporarily disables native packed/jagged execution and restores the model's settings afterward. Learned weights are unchanged.

Attention-24 export traces at opset 20, converts standard operators to opset 24, and replaces typed attention placeholders. The final graph contains no custom MLPF operators. ELU is decomposed into standard operations to avoid CPU placement of `Elu-22` in the CUDA runtime. Decomposed FP16 attention uses power-of-two query scaling to reduce dot-product overflow risk; numerical validation is still required.

## Check a checkpoint on CPU

Keep the checkpoint, its `model_kwargs.pkl`, and a compatible TFDS dataset together. Use the [CPU ONNX environment](../getting-started/installation.md). For a CLD attention checkpoint trained on the [downloaded 3.3.0 data](../datasets/download.md):

```bash
uv run --project envs/ort-cpu python3 scripts/cms-validate-onnx.py \
  --checkpoint /path/to/experiment/checkpoints/checkpoint-2000.pth \
  --model-kwargs /path/to/experiment/model_kwargs.pkl \
  --dataset cld_edm_ttbar_pf/1:3.3.0 \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --num-events 10 --batch-size 2 \
  --device cpu --num-threads 1 \
  --configs PT_ATTN_MATH_FP32 ONNX_ATTN_MATH_FP32 \
  --save-inputs --profile \
  --outdir onnx_validation/cld-cpu
```

Success creates `summary.json`, ONNX files, saved `inputs_*.npz` batches, and diagnostic plots. Inspect exceptions, non-finite outputs, numerical differences, and padding before interpreting timing. The first configuration is the numerical reference; keep the FP32 PyTorch reference first.

Each batch contains distinct events; the final batch can be smaller. Export tracing uses two unequal-length events independently of inference batch size. Summary entries describe one batch invocation: `runtime` is seconds per batch, `size` counts valid elements, and `lengths`, `batch_size`, `event_indices`, and `padded_length` describe the workload.

## Qualify GPU precision and placement

The validator's attention configurations are:

| Configuration | Computation |
|---|---|
| `PT_ATTN_MATH_FP32` / `FP16` | PyTorch dense masked math attention |
| `PT_ATTN_FLASH_FP16` | Packed variable-length PyTorch FlashAttention |
| `ONNX_ATTN_MATH_FP32` / `FP16` | Decomposed attention, full model at the named precision |
| `ONNX_ATTN_MATH_FP32_FP16` | FP32 model with FP16 decomposed attention |
| `ONNX_ATTN_OP_FP32` | FP32 Attention-24; no FlashAttention claim |
| `ONNX_ATTN_FLASH_FP32_FP16` | FP32 model with FP16 Attention-24 |
| `ONNX_ATTN_FLASH_FP16` | Full FP16 Attention-24 model |

On a compatible NVIDIA GPU:

```bash
ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 \
uv run --project envs/ort-gpu python3 scripts/cms-validate-onnx.py \
  --checkpoint /path/to/experiment/checkpoints/checkpoint-2000.pth \
  --model-kwargs /path/to/experiment/model_kwargs.pkl \
  --dataset cld_edm_ttbar_pf/1:3.3.0 \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --num-events 12 --batch-size 4 --pad-bin-size 512 --device cuda \
  --configs PT_ATTN_MATH_FP32 PT_ATTN_FLASH_FP16 ONNX_ATTN_FLASH_FP16 \
  --save-inputs --profile --require-full-gpu \
  --outdir onnx_validation/cld-cuda
```

Flash configurations require CUDA in this validator. Native PyTorch Flash packs valid elements; ONNX keeps padded tensors and passes valid lengths. ONNX Flash runs request `sdpa_kernel=1`; verify `SdpaKernel=FLASH_ATTENTION` in runtime diagnostics. A CUDA provider profile establishes placement, while kernel evidence establishes fusion. `--require-full-gpu` rejects CPU compute fallback.

FP16 is a separate numerical qualification. The retained trained-checkpoint evidence passes the portable FP32 matrix but fails the stated FP32 tolerance for both optimized FP16 profiles. Review [Export qualification and evidence](export-evidence.md) before choosing tolerances or making performance claims.

## Replay on AMD or another runtime

The inference-only `envs/ort-amd` project installs ORT 1.29 and AMD's MIGraphX plugin for a ROCm 10.1 host. Start with decomposed attention and saved input batches:

```bash
uv run --project envs/ort-amd python3 scripts/validate-onnx-runtime.py \
  --model onnx_validation/cld-cpu/model_math_fp32.onnx \
  --inputs onnx_validation/cld-cpu/inputs_*.npz \
  --provider migraphx --require-full-gpu \
  --outdir onnx_validation/cld-amd
```

The replay tool compares valid outputs with ORT CPU, rejects non-finite or out-of-tolerance results, and records first-call timing, repeated-call timing, and provider profiles. It registers the plugin when needed and rejects whole-session fallback. The same tool accepts `--provider cpu` or `cuda` in their corresponding environments.

The packaged AMD plugin environment and the runtime used for retained W7900 evidence are different stacks. The evidence uses native ORT MIGraphX 1.23.2 with MIGraphX 2.18 and ROCm SDK libraries; its exact setup is in [the GPU tooling guide](../../ngt/README.md). Installation of the packaged plugin does not establish equivalent execution or fusion. Attention-24 support on MIGraphX is not qualified by the retained results; AMD optimized attention uses the decomposed FP16 representation.

Dynamic graphs may require CPU shape operations. Static specialization is available in the qualification tools for strict GPU placement and bounded compilation cost. Record runtime versions and loaded libraries with every result.

(heptv2-deterministic-hashing)=
## HEPTv2 and checkpoint compatibility

HEPTv2 defaults to deterministic hashing (`model.heptv2.deterministic_hashing: true`). It quantizes eta/phi and projection coefficients, uses int64 arithmetic, and resolves equal sort keys by input order. Export uses exact integer inverse permutations and separate materialized output heads. Pad input lengths to a multiple of the model's block size.

Deterministic coordinates and bucket assignments can change predictions from checkpoints trained with floating-point hashing. Setting the option to `false` selects floating-point hashing but does not guarantee historical tie ordering or agreement across runtimes. Likewise, older attention checkpoints trained with unmasked padding require validation under the current masking semantics.

The qualified HEPTv2 export profile is portable FP32. Integer-preserving compiler support matters: the older MIGraphX 2.15 stack lowers hashing arithmetic to FP32 and is unsuitable for this profile. The [qualification protocol](export-evidence.md) describes static GPU execution, compiler settings, real-event references, and the limits of the retained evidence. GNNLSH and HEPT have separate validator configurations; inspect `--help` for the available names. LitePT remains NVIDIA-oriented.

## Regression coverage and benchmarking

The [attention export CI workflow](https://github.com/jpata/particleflow/blob/main/.github/workflows/attention-export.yml) uses a lightweight ORT 1.30 CPU environment. The user-facing CPU/CUDA projects pin ORT 1.29. The CI suite checks graph validity, changed batch/sequence dimensions, padding isolation, zero padded outputs, precision behavior, deterministic sorting, and output routing:

```bash
mkdir -p onnx_validation
PYTHONPATH=. python -m pytest \
  tests/test_attention24_export.py tests/test_attention_export_profiles.py \
  tests/test_heptv2_export.py tests/test_attention_runtime.py \
  -q --basetemp=onnx_validation/export-ci
```

GPU and offline-compiler checks are opt-in and need their specified environments. CPU CI does not qualify GPU kernels. See [Export qualification and evidence](export-evidence.md) for those commands and acceptance criteria.

For attention-only scaling, use `scripts/benchmark-sdpa-onnx.py` and `scripts/plot-sdpa-onnx-summary.py`. For example, in the CPU environment:

```bash
uv run --project envs/ort-cpu python3 scripts/benchmark-sdpa-onnx.py \
  --device cpu --batch-sizes 1 2 4 --seq-lens 512 1280 \
  --length-policy balanced --modes masked --mask-fraction 0.5 \
  --num-heads 16 --head-dim 16 --num-threads 1 --warmup 3 --iters 10 \
  --variants pt_math_fp32 ort_attention24_fp32 \
  --outdir onnx_validation/batch-cpu
```

Success writes summary JSON/CSV with workload and timing data. `--export-shapes both` compares dynamic graphs with fixed-shape exports. Full-model measurements use `scripts/benchmark-model-export.py` and the qualification protocol. Report warm-up, precision, valid and padded sizes, batch policy, and whether transfers and compilation are included. ORT `session.run` timings include host/device transfers; PyTorch GPU inputs may already reside on the device. PyTorch allocator statistics omit ORT allocations, so use process RSS and device-memory sampling for complete-stack comparisons.
