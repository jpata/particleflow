import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('resume_manifest', Path(__file__).resolve().parents[1] / 'scripts/training/resume_manifest.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_run(tmp_path):
    manifest = tmp_path / 'scenario-manifest.json'
    manifest.write_text(json.dumps({
        'command': ['python3', 'pipeline.py', 'train', '--load', 'old.pth'],
        'resolved_config': {'num_steps': 100},
        'platform': {'environment': {'OMP_NUM_THREADS': '4'}},
    }))
    (tmp_path / 'checkpoints').mkdir()
    return manifest


def test_resume_selects_numeric_latest_at_each_launch(tmp_path):
    manifest = make_run(tmp_path)
    (tmp_path / 'checkpoints/checkpoint-9.pth').touch()
    with patch.object(runner.subprocess, 'run') as launch:
        runner.resume_manifest(manifest)
        assert launch.call_args.kwargs['env']['OMP_NUM_THREADS'] == '4'
        assert launch.call_args.args[0][-1].endswith('checkpoint-9.pth')
        (tmp_path / 'checkpoints/checkpoint-20.pth').touch()
        runner.resume_manifest(manifest)
        assert launch.call_args.args[0][-1].endswith('checkpoint-20.pth')
        (tmp_path / 'checkpoints/checkpoint-100.pth').touch()
        runner.resume_manifest(manifest)
        assert launch.call_count == 2


def test_resume_lock_and_failure_propagation(tmp_path):
    manifest = make_run(tmp_path)
    with (tmp_path / '.training.lock').open('a') as lock:
        runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another training'):
            runner.resume_manifest(manifest)
    with patch.object(runner.subprocess, 'run', side_effect=runner.subprocess.CalledProcessError(1, 'train')):
        with pytest.raises(runner.subprocess.CalledProcessError):
            runner.resume_manifest(manifest)


def test_resume_starts_without_checkpoint_and_expands_job_environment(tmp_path, monkeypatch):
    manifest = make_run(tmp_path)
    data = json.loads(manifest.read_text())
    data['command'] = ['python3', 'pipeline.py', 'train']
    data['platform']['environment']['CACHE'] = '/tmp/cache-${SLURM_JOB_ID}'
    manifest.write_text(json.dumps(data))
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    with patch.object(runner.subprocess, 'run') as launch:
        runner.resume_manifest(manifest)
        assert '--load' not in launch.call_args.args[0]
        assert launch.call_args.kwargs['env']['CACHE'] == '/tmp/cache-12345'
        (tmp_path / 'checkpoints/checkpoint-5.pth').touch()
        runner.resume_manifest(manifest)
        assert launch.call_args.args[0][-2] == '--load'
        assert launch.call_args.args[0][-1].endswith('checkpoint-5.pth')
