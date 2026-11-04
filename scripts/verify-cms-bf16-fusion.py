"""Check BF16 fusion evidence against the exact graphs used for full qualification."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def text(path):
    return path.read_text() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()).decode()


def verify(root):
    cases = json.loads((root / "input-manifest.json").read_text())["cases"]
    buckets = sorted({c["slots"] for c in cases})
    proofs = []
    for model in ["attention", "heptv2"]:
        for provider, profile in [("cuda", "attention24-bf16"), ("migraphx", "decomposed-bf16")]:
            for fusion in ["on", "off"]:
                results = root / "results" / model / provider / profile / fusion
                for slots in buckets:
                    record = json.loads((results / f"bucket_{slots}.json").read_text())
                    assert record["placement"]["full_provider_placement"]
                    assert len(record["cases"]) == sum(c["slots"] == slots for c in cases)
                    proof = dict(model=model, provider=provider, profile=profile, fusion=fusion, slots=slots, static_sha256=record["static_sha256"])
                    if provider == "cuda":
                        diagnostic = root / "fusion-proof" / model / provider / fusion / str(slots)
                        metadata = json.loads((diagnostic / f"bucket_{slots}.json").read_text())
                        assert metadata["static_sha256"] == record["static_sha256"]
                        assert metadata["placement"]["full_provider_placement"]
                        log = text(diagnostic / "run.log")
                        dispatches = [line for line in log.splitlines() if "Operator=Attention " in line]
                        selected = "FLASH_ATTENTION" if fusion == "on" else "MATH"
                        assert len(dispatches) == 3 * (metadata["timing"]["warmup_calls"] + len(metadata["cases"]))
                        assert all("DataType=bf16" in line and f"SdpaKernel={selected}" in line for line in dispatches)
                        proof.update(dispatch_count=len(dispatches), selected=selected, evidence=str(diagnostic.relative_to(root) / "run.log"))
                    else:
                        log = text(results / f"bucket_{slots}.log")
                        assert 'module: "main"\n' in log
                        final = 'module: "main"\n' + log.rsplit('module: "main"\n', 1)[1].split("\n\n", 1)[0] + "\n"
                        objects = [
                            line
                            for line in final.splitlines()
                            if "gpu::code_object" in line
                            and "symbol_name=mlir_" in line
                            and "reduce_max" in line
                            and "reduce_sum" in line
                            and line.count("dot") >= 2
                        ]
                        assert len(objects) == (3 if fusion == "on" else 0), (model, fusion, slots, objects)
                        assert all("bf16_type" in line for line in objects)
                        quadratic = [
                            line for line in final.splitlines() if ("allocate" in line or "= load[" in line) and f", {slots}, {slots}" in line
                        ]
                        if model == "attention":
                            assert not quadratic if fusion == "on" else bool(quadratic)
                        program = results / f"bucket_{slots}.compiled-program.txt"
                        program.write_text(final)
                        proof.update(
                            fused_attention_objects=len(objects),
                            global_quadratic_buffers=quadratic,
                            compiled_program_sha256=hashlib.sha256(final.encode()).hexdigest(),
                            evidence=str(program.relative_to(root)),
                        )
                    proof["verified"] = True
                    proofs.append(proof)
    result = dict(
        verified_shapes=len(proofs),
        expected_shapes=8 * len(buckets),
        proofs=proofs,
        heptv2_limitation="Only BF16 attention output is fused; separate FP32 QK/log-sum-exp branch remains for bucket merging.",
    )
    (root / "fusion-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"verified_shapes": verify(args.root)["verified_shapes"]}))
