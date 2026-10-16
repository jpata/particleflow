"""Numerical contract checks for the safe rewrite and dynamic export."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from study import Attention, errors, inputs, native
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_safe_rewrite_preserves_padding_empty_and_holey_masks(self):
        for dtype in [np.float32, np.float16]:
            for mode in ["unmasked", "padding", "empty", "holes"]:
                with self.subTest(dtype=dtype, mode=mode):
                    arrays = inputs(2, 16, 2, 16, dtype, mode)
                    reference = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(arrays)]).numpy()
                    output = native(arrays, explicit=True).numpy()
                    self.assertTrue(errors(output, reference, dtype == np.float16)["correct"])
                    if mode == "empty":
                        np.testing.assert_array_equal(output[0], 0)

    def test_exported_safe_graph_preserves_semantics_at_new_sizes(self):
        arrays = inputs(2, 16, 2, 16, np.float16, "padding")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safe.onnx"
            axes = {name: {0: "batch", 2: "seq"} for name in ["query", "key", "value", "output"]}
            axes["mask"] = {0: "batch", 1: "seq"}
            with sdpa_kernel(SDPBackend.MATH):
                torch.onnx.export(
                    Attention("padding", explicit=True),
                    tuple(torch.from_numpy(x) for x in arrays),
                    str(path),
                    input_names=["query", "key", "value", "mask"],
                    output_names=["output"],
                    opset_version=20,
                    dynamo=False,
                    dynamic_axes=axes,
                )
            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.assertNotIn("IsNaN", [n.op_type for n in model.graph.node])
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            for seq in [16, 32, 16]:
                for mode in ["empty", "holes", "padding"]:
                    with self.subTest(seq=seq, mode=mode):
                        arrays = inputs(2, seq, 2, 16, np.float16, mode)
                        reference = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(arrays)]).numpy()
                        output = session.run(None, dict(zip(["query", "key", "value", "mask"], arrays)))[0]
                        self.assertTrue(errors(output, reference, True)["correct"])
                        if mode == "empty":
                            np.testing.assert_array_equal(output[0], 0)

    def test_batch_sweep_holds_average_padding_fraction(self):
        for batch in [1, 2, 4, 8, 16]:
            mask = inputs(batch, 512, 2, 16, np.float16, "padding", balanced=True)[3]
            self.assertAlmostEqual(float(mask.mean()), 0.75, places=3)


if __name__ == "__main__":
    unittest.main()
