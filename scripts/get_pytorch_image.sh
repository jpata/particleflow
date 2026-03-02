#!/bin/bash
# Extracts the PyTorch container image from particleflow_spec.yaml
grep -A 5 "^project:" particleflow_spec.yaml | grep "container:" | head -n 1 | awk '{print $2}' | tr -d '"'
