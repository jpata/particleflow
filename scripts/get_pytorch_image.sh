#!/bin/bash
# Extracts the PyTorch container image from particleflow_spec.yaml
python3 -c "import yaml; print(yaml.safe_load(open('particleflow_spec.yaml'))['project'].get('container', ''))"
