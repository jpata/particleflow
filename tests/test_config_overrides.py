import unittest
import argparse
import yaml
import tempfile
import os
from mlpf.conf import MLPFConfig


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
                        "type": "gnn_lsh",
                        "gnn_lsh": {
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

    def test_build_config_from_spec(self):
        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod")
        config = config_obj.model_dump()

        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["num_steps"], 100)
        self.assertEqual(config["lr"], 0.001)
        self.assertEqual(config["conv_type"], "gnn_lsh")
        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 2)
        self.assertEqual(config["dataset"], "cms")
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

        extra_args = ["--model.gnn_lsh.width", "128", "--num_steps", "200"]

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args, extra_args=extra_args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnn_lsh"]["width"], 128)
        self.assertEqual(config["num_steps"], 200)

    def test_override_config_convenience_flags(self):
        args = argparse.Namespace()
        args.num_convs = 5
        args.train = True
        args.test_datasets = []

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 5)

    def test_pipeline_overrides(self):
        args = argparse.Namespace()
        args.pipeline = True
        args.command = "train"
        args.test_datasets = []

        config_obj = MLPFConfig.from_spec(self.temp_spec.name, "test_model", "test_prod", args=args)
        config = config_obj.model_dump()

        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 1)
        self.assertEqual(config["model"]["gnn_lsh"]["width"], 32)
        self.assertEqual(config["train_dataset"]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"], ["10"])
        self.assertEqual(config["test_dataset"]["cms_pf_ttbar"]["splits"], ["10"])


if __name__ == "__main__":
    unittest.main()
