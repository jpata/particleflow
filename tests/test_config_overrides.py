"""
Spec: Validates the 'MLPFConfig' class and its 'from_spec' method. Tests configuration construction from YAML specs with recursive path resolution ('resolve_path'). Key scenarios: Argparse Namespace overrides, extra CLI flags using dot-notation (e.g., '--model.gnnlsh.width'), pipeline-specific overrides (like CI/CD splits), and error handling for invalid or forbidden field overrides.
"""

import unittest
import argparse
from dataclasses import FrozenInstanceError
import yaml
import tempfile
import os
from mlpf.conf import AttentionType, MLPFConfig, ModelType, Dataset, _PIPELINE_DATASETS


class TestConfigOverrides(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "project": {
                "workspace_dir": "/tmp/particleflow",
            },
            "models": {
                "defaults": {
                    "batch_size": 32,
                    "num_steps": 100,
                },
                "test_model": {
                    "architecture": {
                        "type": "gnnlsh",
                        "gnnlsh": {
                            "num_convs": 2,
                            "width": 64,
                        },
                    },
                    "hyperparameters": {
                        "lr": 0.001,
                    },
                    "train_datasets": {"physical_pu": {"samples": [{"name": "cms_pf_ttbar", "version": "1.0.0", "splits": ["1"]}]}},
                    "validation_datasets": {"physical_pu": {"samples": [{"name": "cms_pf_ttbar", "version": "1.0.0", "splits": ["1"]}]}},
                    "test_datasets": [{"name": "cms_pf_ttbar", "version": "1.0.0"}],
                },
            },
            "productions": {
                "test_prod": {
                    "type": "cms",
                    "workspace_dir": "${project.workspace_dir}/test_prod",
                }
            },
        }
        self.temp_spec = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(self.spec, self.temp_spec)
        self.temp_spec.close()

    def tearDown(self):
        os.unlink(self.temp_spec.name)

    def write_spec(self):
        with open(self.temp_spec.name, "w") as handle:
            yaml.safe_dump(self.spec, handle)

    def test_build_config_from_spec(self):
        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod")
        config = config_obj.model_dump()

        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["num_steps"], 100)
        self.assertEqual(config["lr"], 0.001)
        self.assertEqual(config["conv_type"], ModelType.GNNLSH)
        self.assertEqual(config["model"]["gnnlsh"]["num_convs"], 2)
        self.assertEqual(config["dataset"], Dataset.CMS)
        self.assertTrue("/tmp/particleflow/test_prod/tfds" in config["data_dir"])

    def test_override_config_basic(self):
        args = argparse.Namespace()
        args.batch_size = 64
        args.lr = 0.01
        args.train = True
        args.test = False
        args.make_plots = True
        args.test_datasets = []

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args)
        config = config_obj.model_dump()

        self.assertEqual(config["batch_size"], 64)
        self.assertEqual(config["lr"], 0.01)
        self.assertTrue(config["train"])
        self.assertFalse(config["test"])
        self.assertTrue(config["make_plots"])

    def test_override_config_extra_args(self):
        args = argparse.Namespace()
        args.train = True
        args.test_datasets = []

        extra_args = ["--model.gnnlsh.width", "128", "--num_steps", "200", "--task_loss_weights.calibration_steps", "20"]

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args, extra_args=extra_args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnnlsh"]["width"], 128)
        self.assertEqual(config["num_steps"], 200)
        self.assertEqual(config["task_loss_weights"]["calibration_steps"], 20)

    def test_override_config_convenience_flags(self):
        args = argparse.Namespace()
        args.num_convs = 5
        args.attention_type = "simple"
        args.train = True
        args.test_datasets = []

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnnlsh"]["num_convs"], 5)
        self.assertEqual(config["model"]["attention"]["attention_type"], AttentionType.SIMPLE)

    def test_data_config_filters_splits_and_preserves_batch_sizes(self):
        datasets = self.spec["models"]["test_model"]
        for key in ["train_datasets", "validation_datasets"]:
            datasets[key]["physical_pu"]["batch_size"] = 8
            datasets[key]["physical_pu"]["samples"][0].update({"splits": ["1", "2"], "batch_size": 3})
        datasets["test_datasets"][0].update({"splits": ["1", "2"], "batch_size": 5})
        self.write_spec()

        config = MLPFConfig.from_spec(
            self.temp_spec.name,
            "test_model",
            "test_prod",
            args=argparse.Namespace(data_config=" 2, missing ", test_datasets=[]),
        )

        self.assertEqual(config.data_config, ["2", "missing"])
        train_sample = config.train_dataset["cms"]["physical_pu"].samples["cms_pf_ttbar"]
        self.assertEqual(config.train_dataset["cms"]["physical_pu"].batch_size, 8)
        self.assertEqual(train_sample.batch_size, 3)
        self.assertEqual(train_sample.splits, ["2"])
        self.assertEqual(config.valid_dataset["cms"]["physical_pu"].samples["cms_pf_ttbar"].splits, ["2"])
        self.assertEqual(config.test_dataset["cms_pf_ttbar"].batch_size, 5)
        self.assertEqual(config.test_dataset["cms_pf_ttbar"].splits, ["2"])

    def test_scalar_data_config_and_explicit_test_selection(self):
        self.spec["models"]["test_model"]["train_datasets"]["physical_pu"]["samples"][0]["splits"] = ["1", "2"]
        self.write_spec()

        config = MLPFConfig.from_spec(
            self.temp_spec.name,
            "test_model",
            "test_prod",
            args=argparse.Namespace(data_config=2, test_datasets=["selected_elsewhere"]),
        )

        self.assertEqual(config.data_config, ["2"])
        self.assertEqual(config.enabled_test_datasets, ["selected_elsewhere"])

    def test_nested_paths_are_resolved_after_overrides(self):
        self.spec["models"]["test_model"]["raytune"] = {"storage_path": "${project.workspace_dir}/ray"}
        self.write_spec()

        config = MLPFConfig.from_spec(
            self.temp_spec.name,
            "test_model",
            "test_prod",
            extra_args=["--load", "${project.workspace_dir}/checkpoint.pt"],
        )

        self.assertEqual(config.raytune["storage_path"], "/tmp/particleflow/ray")
        self.assertEqual(config.load, "/tmp/particleflow/checkpoint.pt")

    def test_cld_pipeline_overrides(self):
        model = self.spec["models"]["test_model"]
        model["dataset"] = "cld"
        model["train_datasets"] = {"physical": {"batch_size": 7, "samples": [{"name": "cld_edm_ttbar_pf", "version": "1.0.0", "splits": ["1"]}]}}
        model["validation_datasets"] = model["train_datasets"]
        model["test_datasets"] = [{"name": "cld_edm_ttbar_pf", "version": "1.0.0"}]
        self.spec["productions"]["test_prod"]["type"] = "cld"
        self.write_spec()

        config = MLPFConfig.from_spec(
            self.temp_spec.name,
            "test_model",
            "test_prod",
            args=argparse.Namespace(pipeline=True, test_datasets=[]),
        )

        self.assertEqual(config.gpu_batch_multiplier, 8)
        self.assertEqual(config.train_dataset["cld"]["physical"].batch_size, 7)
        self.assertEqual(config.train_dataset["cld"]["physical"].samples["cld_edm_ttbar_pf"].splits, ["10"])
        self.assertEqual(config.valid_dataset["cld"]["physical"].samples["cld_edm_ttbar_pf"].version, "3.2.1")
        self.assertEqual(config.test_dataset["cld_edm_ttbar_pf"].splits, ["10"])

    def test_pipeline_dataset_overrides_are_immutable_named_records(self):
        cld_override = _PIPELINE_DATASETS["cld"]

        self.assertEqual(cld_override.physical_name, "physical")
        self.assertEqual(cld_override.sample_name, "cld_edm_ttbar_pf")
        self.assertEqual(cld_override.version, "3.2.1")
        self.assertEqual(cld_override.gpu_batch_multiplier, 8)
        self.assertIsNone(_PIPELINE_DATASETS["cms"].gpu_batch_multiplier)
        with self.assertRaises(FrozenInstanceError):
            cld_override.version = "changed"

    def test_pipeline_overrides(self):
        args = argparse.Namespace()
        args.pipeline = True
        args.command = "train"
        args.test_datasets = []

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnnlsh"]["num_convs"], 1)
        self.assertEqual(config["model"]["gnnlsh"]["width"], 32)
        self.assertEqual(config["train_dataset"]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"], ["10"])
        self.assertEqual(config["test_dataset"]["cms_pf_ttbar"]["splits"], ["10"])

    def test_invalid_extra_args(self):
        args = argparse.Namespace()
        args.train = True
        args.test_datasets = []

        # "invalid_arg" is not a flag (doesn't start with --) and not part of a key=value pair
        extra_args = ["--num_steps", "200", "invalid_arg"]

        with self.assertRaisesRegex(ValueError, "Could not parse extra argument: invalid_arg"):
            MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args, extra_args=extra_args)

    def test_extra_forbidden_args(self):
        args = argparse.Namespace()
        args.train = True
        args.test_datasets = []

        # This should fail during Pydantic validation
        extra_args = ["--extra_forbidden_field", "value"]

        with self.assertRaises(Exception):
            MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args, extra_args=extra_args)

    def test_invalid_production_name(self):
        with self.assertRaisesRegex(ValueError, "Production missing not found in spec"):
            MLPFConfig.from_spec(self.temp_spec.name, "test_model", "missing")


if __name__ == "__main__":
    unittest.main()
