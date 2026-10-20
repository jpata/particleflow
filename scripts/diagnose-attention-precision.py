"""Separate trained FP32 drift from export error against each FP16 graph's PyTorch contract."""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def compare(actual, expected, valid):
    result = {}
    for name, value, target in zip(['bid', 'id', 'momentum', 'pu'], actual, expected, strict=True):
        a, b = value[valid].astype(np.float64), target[valid].astype(np.float64)
        delta = abs(a - b)
        result[name] = {'max_abs': float(delta.max()), 'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            'close': bool(np.allclose(a, b, atol=1e-3, rtol=3e-3)), 'violating_elements': int(np.count_nonzero(delta > 1e-3 + 3e-3 * abs(b))),
            'argmax_disagreements': int(np.count_nonzero(a.argmax(-1) != b.argmax(-1))) if name in ['bid', 'id', 'pu'] else None}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--case', default='cms_pf_qcd_1_0000.npz')
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    import torch
    from mlpf.conf import MLPFConfig
    from mlpf.model.mlpf import MLPF
    from mlpf.model.onnx_attention import _export_settings
    from mlpf.model.onnx_runtime import create_session
    spec = importlib.util.spec_from_file_location('qualification', Path(__file__).with_name('qualify-heptv2-onnx.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((a.root / 'attention/manifest.json').read_text())
    config = MLPFConfig.model_validate(manifest['config'])
    model = MLPF(config).float().eval()
    state = torch.load(manifest['checkpoint']['path'], map_location='cpu', weights_only=True)['model_state_dict']
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('task_loss_weighter.')}, strict=True)
    model = model.cuda()
    with np.load(a.root / 'attention' / a.case) as data:
        feeds = {key: data[key].copy() for key in ['Xfeat_normed', 'mask']}
        fp32 = [data[key].copy() for key in ['bid', 'id', 'momentum', 'pu']]
    valid = feeds['mask'].astype(bool)
    a.outdir.mkdir(parents=True, exist_ok=True)
    report = {'case': a.case, 'provider': 'cuda', 'machine': 'L40S', 'checkpoint': manifest['checkpoint'], 'reference': 'PyTorch CUDA evaluates the exact per-profile export operations, with FP16 attention internals and FP32 surrounding layers', 'profiles': {}}
    for name, format in [('attention24-fp16', 'attention24'), ('attention-decomposed-fp16', 'decomposed')]:
        x, mask = (torch.from_numpy(feeds[key]).cuda() for key in ['Xfeat_normed', 'mask'])
        with torch.inference_mode(), _export_settings(model, format, torch.float16), patch('torch.onnx.is_in_onnx_export', return_value=True):
            contract = [value.cpu().numpy() for value in model(x, mask)]
        del x, mask
        torch.cuda.empty_cache()
        path = a.outdir / (name + '.onnx')
        helper.specialize_model(a.root / name / 'model.onnx', path, feeds)
        session = create_session(path, 'cuda', require_full_gpu=True, flash=format == 'attention24')
        actual = session.run(['bid', 'id', 'momentum', 'pu'], feeds)
        del session
        report['profiles'][name] = {'onnx_vs_fp32': compare(actual, fp32, valid),
            'torch_fp16_contract_vs_fp32': compare(contract, fp32, valid), 'onnx_vs_torch_fp16_contract': compare(actual, contract, valid)}
        np.savez(a.outdir / (name + '.npz'), **{f'{kind}_{key}': value for kind, values in [('torch', contract), ('onnx', actual)] for key, value in zip(['bid', 'id', 'momentum', 'pu'], values, strict=True)})
        print(name, report['profiles'][name], flush=True)
    (a.outdir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
