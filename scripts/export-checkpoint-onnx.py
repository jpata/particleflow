"""Export a trained checkpoint and independent FP32 references on real TFDS events."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['cms_pf_ttbar/1:3.0.0', 'cms_pf_qcd/1:3.0.0', 'cms_pf_ztt/1:3.0.0'])
    p.add_argument('--split', default='test')
    p.add_argument('--num-events', type=int, default=2, help='Actual events per dataset, without truncation or repetition')
    p.add_argument('--attention-format', choices=['decomposed', 'attention24'], default='decomposed')
    p.add_argument('--attention-dtype', choices=['float32', 'float16'], default='float32')
    a = p.parse_args()
    if a.num_events < 1:
        p.error('--num-events must be positive')
    import torch
    import tensorflow_datasets as tfds
    from mlpf.conf import MLPFConfig, AttentionType
    from mlpf.model.mlpf import MLPF
    from mlpf.model.onnx_attention import export_mlpf

    torch.set_num_threads(1)
    checkpoint = a.checkpoint or max((a.experiment / 'checkpoints').glob('checkpoint-*.pth'), key=lambda path: int(path.stem.split('-')[-1]))
    kwargs = a.experiment / 'model_kwargs.pkl'
    with kwargs.open('rb') as f:
        saved = pickle.load(f)
    config = MLPFConfig.model_validate(saved if isinstance(saved, dict) else saved.model_dump())
    training_config = config.model_dump(mode='json')
    if config.dataset.value != 'cms' or config.model.output_mode.value != 'elementwise':
        p.error('This qualification requires an elementwise CMS checkpoint')
    model_type = config.model.type.value
    if model_type == 'heptv2' and (a.attention_format != 'decomposed' or a.attention_dtype != 'float32'):
        p.error('HEPTv2 uses the portable FP32 profile')
    reference_overrides = {}
    if config.model.attention is not None:
        config.model.attention.attention_type = AttentionType.MATH
        config.model.attention.use_jagged_attention = False
        config.model.attention.use_flash_attn_varlen = False
        reference_overrides = {'attention_type': 'math', 'use_jagged_attention': False, 'use_flash_attn_varlen': False}
    model = MLPF(config).float().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = state['model_state_dict']
    omitted = [key for key in weights if key.startswith('task_loss_weighter.')]
    weights = {key: value for key, value in weights.items() if key not in omitted}
    model.load_state_dict(weights, strict=True)
    block = max(config.pad_to_multiple_elements or 1, config.model.heptv2.block_size if model_type == 'heptv2' else 1)
    events, dataset_sources = [], []
    for dataset in a.datasets:
        name_config, version = dataset.split(':')
        name, builder_config = name_config.split('/')
        directory = a.data_dir / name / builder_config / version
        builder = tfds.builder_from_directory(str(directory))
        data = builder.as_data_source(split=a.split)
        if len(data) < a.num_events:
            raise ValueError(f'{dataset} has only {len(data)} events in {a.split}')
        source = {'dataset': dataset, 'split': a.split, 'directory': str(directory),
                  'dataset_info_sha256': digest(directory / 'dataset_info.json'), 'indices': list(range(a.num_events))}
        dataset_sources.append(source)
        for index in source['indices']:
            x = torch.as_tensor(np.array(data[index]['X'], copy=True), dtype=torch.float32)
            raw_shape = list(x.shape)
            x = x[x[:, 0] != 0]
            if not len(x) or x.shape[1] > config.input_dim:
                raise ValueError(f'Invalid event shape {x.shape} in {dataset}:{index}')
            x = torch.nn.functional.pad(x, (0, config.input_dim - x.shape[1]))
            if config.sort_data:
                x = x[torch.argsort(x[:, 1], descending=True)]
            if not torch.isfinite(x).all():
                raise ValueError(f'Nonfinite source input in {dataset}:{index}')
            events.append((f'{name}_{builder_config}_{index:04d}', x,
                {'dataset': dataset, 'split': a.split, 'index': index, 'raw_shape': raw_shape}))
    cases = [(name, [x], [source]) for name, x, source in events]
    mixed = [events[i * a.num_events] for i in range(len(a.datasets))]
    cases.append(('mixed_real_batch', [x for _, x, _ in mixed], [source for _, _, source in mixed]))
    a.bundle.mkdir(parents=True, exist_ok=True)
    manifest = {'scope': f'Trained {model_type} export qualification on actual TFDS events; training and qualification datasets are recorded separately',
        'seed': config.seed, 'torch': torch.__version__, 'config': config.model_dump(mode='json'),
        'training_config': training_config, 'reference_overrides': reference_overrides,
        'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': digest(checkpoint), 'step': int(checkpoint.stem.split('-')[-1]),
            'model_kwargs_path': str(kwargs.resolve()), 'model_kwargs_sha256': digest(kwargs), 'strict_weight_loading': True,
            'omitted_training_only_keys': omitted},
        'dataset_sources': dataset_sources,
        'input_preprocessing': {'source': 'raw TFDS X features', 'remove_zero_element_type': True, 'feature_dim': config.input_dim,
            'sort_pt_descending': config.sort_data, 'pad_to_multiple': block, 'clamp': [-60000, 60000],
            'cms_ho_pt_correction': False, 'target_transformations': False},
        'export_profile': {'attention_format': a.attention_format, 'attention_dtype': a.attention_dtype}, 'cases': []}
    outputs = ['bid', 'id', 'momentum', 'pu']
    for index, (name, batch, sources) in enumerate(cases):
        lengths = [len(x) for x in batch]
        size = math.ceil(max(lengths) / block) * block
        x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        x = torch.nn.functional.pad(x, (0, 0, 0, size - x.shape[1])).clamp(-60000, 60000).contiguous()
        mask = (torch.arange(size)[None, :] < torch.tensor(lengths)[:, None]).float()
        if index == 0:
            export_mlpf(model, x, mask, a.bundle / 'model.onnx', attention_format=a.attention_format,
                attention_dtype=torch.float16 if a.attention_dtype == 'float16' else None)
            manifest['model_sha256'] = digest(a.bundle / 'model.onnx')
        with torch.no_grad():
            reference = model(x, mask)
        for key, value in zip(outputs, reference, strict=True):
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite PyTorch reference in {name}/{key}')
        path = a.bundle / (name + '.npz')
        np.savez(path, Xfeat_normed=x.numpy(), mask=mask.numpy(), **{key: value.numpy() for key, value in zip(outputs, reference, strict=True)})
        manifest['cases'].append({'file': path.name, 'sha256': digest(path), 'shape': list(x.shape), 'lengths': lengths, 'sources': sources})
        print(name, list(x.shape), lengths, flush=True)
    (a.bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Exported', checkpoint, a.bundle, flush=True)


if __name__ == '__main__':
    main()
