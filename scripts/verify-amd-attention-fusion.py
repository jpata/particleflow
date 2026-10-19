"""Verify retained MIGraphX compiler programs for every optimized attention shape."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--bundle-root", type=Path, help="Defaults to the evidence root")
    p.add_argument("--compress-logs", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    profile = "attention-decomposed-fp16"
    bundle = (a.bundle_root or a.root) / profile
    manifest = json.loads((bundle / "manifest.json").read_text())
    import onnx

    expected = sum(n.op_type == "Softmax" for n in onnx.load(bundle / "model.onnx").graph.node)
    assert expected > 0
    proofs, missing = [], []
    for case in manifest["cases"]:
        dest = a.root / "supplementary" / profile / Path(case["file"]).stem / "placement"
        if not (dest / "result.json").exists() or not (dest / "exit-code.txt").exists() or (dest / "exit-code.txt").read_text().strip() != "0":
            missing.append(case["file"])
            continue
        log = dest / "run.log"
        text = log.read_text() if log.exists() else gzip.decompress(log.with_suffix(".log.gz").read_bytes()).decode()
        assert 'module: "main"\n' in text
        final = 'module: "main"\n' + text.rsplit('module: "main"\n', 1)[1].split("\n\n", 1)[0] + "\n"
        objects = [
            line
            for line in final.splitlines()
            if "gpu::code_object" in line and "symbol_name=mlir_" in line and "reduce_max" in line and "reduce_sum" in line and line.count("dot") >= 2
        ]
        size = case["shape"][1]
        quadratic = [line for line in final.splitlines() if "allocate" in line and f", {size}, {size}" in line]
        assert len(objects) == expected, (case["file"], len(objects), expected)
        assert not quadratic, (case["file"], quadratic)
        static = dest / "static.onnx"
        proof = {
            "model_sha256": manifest["model_sha256"],
            "case": case["file"],
            "verified": True,
            "expected_attention_operations": expected,
            "fused_attention_code_objects": len(objects),
            "quadratic_attention_allocations": quadratic,
            "static_model_sha256": hashlib.sha256(static.read_bytes()).hexdigest(),
            "program_sha256": hashlib.sha256(final.encode()).hexdigest(),
            "kernel_symbols": [line.split("symbol_name=", 1)[1].split(",global=", 1)[0] for line in objects],
        }
        (dest / "compiled-program.txt").write_text(final)
        (dest / "fusion-verification.json").write_text(json.dumps(proof, indent=2) + "\n")
        proofs.append(proof)
        if a.compress_logs and log.exists():
            log.with_suffix(".log.gz").write_bytes(gzip.compress(log.read_bytes()))
            log.unlink()
    summary = {"verified_cases": len(proofs), "expected_cases": len(manifest["cases"]), "missing": missing, "cases": proofs}
    (a.root / "amd-fusion-verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"verified": len(proofs), "missing": missing}))
    if a.require_complete and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
