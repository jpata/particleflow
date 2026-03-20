import datetime
import logging
import platform
import re
import yaml
from pathlib import Path
from comet_ml import OfflineExperiment, Experiment  # isort:skip


import os


def load_spec(spec_file):
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)

    # Runtime site override via environment variable
    site_override = os.environ.get("PF_SITE")
    if site_override and "project" in spec and "sites" in spec["project"]:
        if site_override in spec["project"]["sites"]:
            # Perform a shallow merge similar to YAML's '<<'
            spec["project"].update(spec["project"]["sites"][site_override])

    return spec


def set_nested_dict(d, key_path, value):
    """Set a value in a nested dictionary using a dot-separated path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    # Try to parse string values as yaml to get correct types (int, float, bool, etc.)
    if isinstance(value, str):
        try:
            # handle cases like "None" -> None, "1" -> 1, "True" -> True
            parsed_value = yaml.safe_load(value)
            # Only use parsed value if it's not a string (unless it was explicitly "None")
            if not isinstance(parsed_value, str) or parsed_value == "None":
                value = parsed_value
        except Exception:
            pass
    d[keys[-1]] = value


def get_nested_dict(d, key_path, default=None):
    """Get a value from a nested dictionary using a dot-separated path."""
    keys = key_path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def resolve_path(path, spec):
    # Recursive substitution for ${...}
    def replace(match):
        key_path = match.group(1).split(".")
        val = spec
        for k in key_path:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
            if val is None:
                return match.group(0)  # fail gracefully
        return str(val)

    prev_path = None
    while path != prev_path:
        prev_path = path
        path = re.sub(r"\$\{(.+?)\}", replace, path)
    return path


def parse_extra_args(extra_args):
    """Parse unrecognized arguments from argparse.parse_known_args()."""
    overrides = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--") and "=" not in extra_args[i + 1]:
                overrides[key] = extra_args[i + 1]
                i += 2
            else:
                overrides[key] = "True"
                i += 1
        elif "=" in arg:
            key, val = arg.split("=", 1)
            overrides[key] = val
            i += 1
        else:
            i += 1
    return overrides


def create_experiment_dir(prefix=None, suffix=None, experiments_dir="experiments"):
    if prefix is None:
        train_dir = Path(experiments_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        train_dir = Path(experiments_dir) / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    if suffix is not None:
        train_dir = train_dir.with_name(train_dir.name + "." + platform.node())

    train_dir.mkdir(parents=True)

    return str(train_dir)


def create_comet_experiment(comet_exp_name, comet_offline=False, outdir=None):
    try:
        if comet_offline:
            logging.info("Using comet-ml OfflineExperiment, saving logs locally.")
            if outdir is None:
                raise ValueError("Please specify am output directory when setting comet_offline to True")

            experiment = OfflineExperiment(
                project_name=comet_exp_name,
                auto_metric_logging=False,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                offline_directory=outdir + "/cometml",
                auto_output_logging="simple",
            )
        else:
            logging.info("Using comet-ml Experiment, streaming logs to www.comet.ml.")

            experiment = Experiment(
                project_name=comet_exp_name,
                auto_metric_logging=False,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                auto_output_logging="simple",
            )
    except Exception as e:
        logging.warning("Failed to initialize comet-ml dashboard: {}".format(e))
        experiment = None
    return experiment
