"""Verify per-shape fusion evidence for the CMS physics qualification sweep."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compress-logs", action="store_true")
    args = parser.parse_args()
    import onnx

    cases = json.loads((args.root / "inputs/manifest.json").read_text())["cases"]
    proofs, missing = [], []
    for provider, profile in [("cuda", "attention24-fp16"), ("migraphx", "decomposed-fp16")]:
        graph = onnx.load(args.root / "models/attention" / profile / "model.onnx")
        expected = sum(node.op_type == ("Attention" if provider == "cuda" else "Softmax") for node in graph.graph.node)
        assert expected > 0
        for fusion in ["on", "off"]:
            root = args.root / "results/attention" / provider / profile / fusion
            for slots in sorted({c["slots"] for c in cases}):
                path = root / f"bucket_{slots}.json"
                if not path.exists():
                    missing.append(str(path.relative_to(args.root)))
                    continue
                metadata = json.loads(path.read_text())
                assert metadata["placement"]["full_provider_placement"]
                assert len(metadata["cases"]) == sum(c["slots"] == slots for c in cases)
                log = root / f"bucket_{slots}.log"
                text = log.read_text() if log.exists() else gzip.decompress(log.with_suffix(".log.gz").read_bytes()).decode()
                proof = {
                    "provider": provider,
                    "profile": profile,
                    "fusion": fusion,
                    "slots": slots,
                    "static_sha256": metadata["static_sha256"],
                    "attention_operations": expected,
                }
                if provider == "cuda":
                    dispatches = [line for line in text.splitlines() if "Operator=Attention " in line]
                    selected = "FLASH_ATTENTION" if fusion == "on" else "MATH"
                    assert len(dispatches) == expected * (len(metadata["cases"]) + 1)
                    assert all(f"SdpaKernel={selected}" in line for line in dispatches)
                    proof.update(dispatch_count=len(dispatches), selected=selected)
                else:
                    assert 'module: "main"\n' in text
                    final = 'module: "main"\n' + text.rsplit('module: "main"\n', 1)[1].split("\n\n", 1)[0] + "\n"
                    objects = [
                        line
                        for line in final.splitlines()
                        if "gpu::code_object" in line
                        and "symbol_name=mlir_" in line
                        and "reduce_max" in line
                        and "reduce_sum" in line
                        and line.count("dot") >= 2
                    ]
                    quadratic = [line for line in final.splitlines() if ("allocate" in line or "= load[" in line) and f", {slots}, {slots}" in line]
                    assert len(objects) == (expected if fusion == "on" else 0)
                    assert not quadratic if fusion == "on" else bool(quadratic)
                    program = root / f"bucket_{slots}.compiled-program.txt"
                    program.write_text(final)
                    proof.update(
                        fused_attention_objects=len(objects),
                        quadratic_buffers=quadratic,
                        compiled_program_sha256=hashlib.sha256(final.encode()).hexdigest(),
                    )
                proof["verified"] = True
                proofs.append(proof)
                if args.compress_logs and log.exists():
                    log.with_suffix(".log.gz").write_bytes(gzip.compress(log.read_bytes()))
                    log.unlink()
    result = {"verified_shapes": len(proofs), "expected_shapes": 4 * len({c["slots"] for c in cases}), "missing": missing, "proofs": proofs}
    (args.root / "fusion-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified_shapes": len(proofs), "missing": missing}), flush=True)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
