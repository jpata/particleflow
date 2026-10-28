"""Explicit provider selection and placement evidence for ONNX qualification."""

import importlib
import json
from collections import Counter
from pathlib import Path

import onnxruntime as ort

PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}


def create_session(path, provider, *, threads=1, profile_prefix=None, require_full_gpu=False, flash=False):
    """Create a requested provider session; never accept whole-session fallback."""
    if require_full_gpu and provider == "cpu":
        raise ValueError("Full GPU qualification requires a GPU provider")
    name = PROVIDERS[provider]
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    if provider == "migraphx":
        # Keep the graph in standard ONNX form for GPU partitioning rather
        # than introducing extended fusions whose support depends on shape.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    if require_full_gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    settings = {"use_tf32": "0"} if provider == "cuda" else {}
    if provider == "cuda":
        ort.preload_dlls()
        if flash:
            settings["sdpa_kernel"] = "1"
    devices = []
    if provider == "migraphx" and hasattr(ort, "get_ep_devices"):
        devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
    if provider == "migraphx" and (devices or name not in ort.get_available_providers()):
        if not devices:
            plugin = importlib.import_module("onnxruntime_ep_migraphx")
            names, libraries = plugin.get_ep_names(), plugin.get_library_paths()
            if not names or len(names) != len(libraries):
                raise RuntimeError("MIGraphX plugin has no consistent provider libraries")
            for plugin_name, library in zip(names, libraries, strict=True):
                ort.register_execution_provider_library(plugin_name, library)
            devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
        if not devices:
            raise RuntimeError("MIGraphX plugin found no GPU device")
        options.add_provider_for_devices([devices[0]], settings)
        session = ort.InferenceSession(str(path), options)
    else:
        if name not in ort.get_available_providers():
            raise RuntimeError(f"Requested provider {name} is unavailable")
        session = ort.InferenceSession(str(path), options, providers=[(name, settings)])
    if name not in session.get_providers():
        raise RuntimeError(f"Requested {name}; session fell back to {session.get_providers()}")
    session.disable_fallback()
    return session


def finish_profile(session, provider, *, require_full_gpu=False):
    """Retain placement evidence and reject missing or CPU compute evidence."""
    path = session.end_profiling()
    counts = Counter(
        event["args"]["provider"]
        for event in json.loads(Path(path).read_text())
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    qualified = bool(counts) and set(counts) == {PROVIDERS[provider]}
    if require_full_gpu and not qualified:
        raise RuntimeError(f"Full GPU placement not established: {dict(counts)}; profile: {path}")
    return {"path": path, "compute_providers": dict(counts), "full_provider_placement": qualified}
