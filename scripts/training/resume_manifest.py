#!/usr/bin/env python3
"""Launch a saved scenario component from its latest checkpoint (Python 3.6+)."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def resume_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    run = manifest_path.parent
    # Keep the lock for the subprocess lifetime; duplicate submissions must fail.
    with (run / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another training process owns {}'.format(run))
        manifest = json.loads(manifest_path.read_text())
        command = list(manifest['command'])
        checkpoints = []
        for path in (run / 'checkpoints').glob('checkpoint-*.pth'):
            match = re.fullmatch(r'checkpoint-(\d+)(?:-.*)?\.pth', path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        step, checkpoint = max(checkpoints, default=(0, None), key=lambda item: item[0])
        if step >= manifest['resolved_config']['num_steps']:
            print('Run already complete at step {}'.format(step), flush=True)
            return
        if checkpoint is not None:
            if '--load' in command:
                command[command.index('--load') + 1] = str(checkpoint)
            else:
                command.extend(['--load', str(checkpoint)])
        print('Starting {} from {}'.format(run, checkpoint or 'scratch'), flush=True)
        environment = os.environ.copy()
        environment.update({key: os.path.expandvars(value)
                            for key, value in manifest['platform'].get('environment', {}).items()})
        subprocess.run(command, check=True, env=environment)


if __name__ == '__main__':
    resume_manifest(sys.argv[1])
