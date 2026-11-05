import pytest


@pytest.fixture
def training_files(tmp_path):
    """Small, self-contained training inputs for orchestration tests."""
    import yaml

    spec_path = tmp_path / "spec.yaml"
    samples = [{"name": "fixture_sample", "version": "1.0.0", "splits": ["fixture"]}]
    spec_path.write_text(
        yaml.safe_dump(
            {
                "project": {"workspace_dir": str(tmp_path)},
                "productions": {"fixture": {"type": "cld", "workspace_dir": str(tmp_path)}},
                "models": {
                    "fixture_model": {
                        "dataset": "cld",
                        "architecture": {"type": "attention", "attention": {"num_convs": 2, "num_heads": 2, "head_dim": 8}},
                        "train_datasets": {"physical": {"batch_size": 2, "samples": samples}},
                        "validation_datasets": {"physical": {"batch_size": 2, "samples": samples}},
                        "test_datasets": samples,
                    }
                },
            }
        )
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "fixture-scenario",
                "spec_file": str(spec_path),
                "production_name": "fixture",
                "variants": {"first": {"model_name": "fixture_model"}, "second": {"model_name": "fixture_model", "overrides": {"lr": 0.002}}},
                "seeds": [17],
                "training": {"global_batch_size": 12, "parameters": {"num_steps": 13, "dtype": "float32", "compile": False}},
                "allowed_variant_differences": ["lr"],
            },
            sort_keys=False,
        )
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "fixture-platform",
                "gpus": 1,
                "data_dir": str(tmp_path / "data"),
                "experiments_dir": str(tmp_path / "runs"),
                "runtime_overrides": {"num_workers": 0},
                "slurm": {"partition": "fixture-partition", "cpus_per_task": 3},
            }
        )
    )
    return scenario_path, profile_path
