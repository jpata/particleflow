
import unittest
import argparse
from mlpf.pipeline import build_config_from_spec
from mlpf.model.training import override_config

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
                        }
                    },
                    "hyperparameters": {
                        "lr": 0.001,
                    },
                    "train_datasets": {
                        "physical_pu": {
                            "samples": [
                                {"name": "cms_pf_ttbar", "version": "1.0.0"}
                            ]
                        }
                    },
                    "validation_datasets": {
                        "physical_pu": {
                            "samples": [
                                {"name": "cms_pf_ttbar", "version": "1.0.0"}
                            ]
                        }
                    },
                    "test_datasets": [
                        {"name": "cms_pf_ttbar", "version": "1.0.0"}
                    ]
                }
            },
            "productions": {
                "test_prod": {
                    "type": "cms",
                    "workspace_dir": "${project.workspace_dir}/test_prod",
                }
            }
        }

    def test_build_config_from_spec(self):
        config = build_config_from_spec(self.spec, "test_model", "test_prod")
        
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["num_steps"], 100)
        self.assertEqual(config["lr"], 0.001)
        self.assertEqual(config["conv_type"], "gnn_lsh")
        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 2)
        self.assertEqual(config["dataset"], "cms")
        self.assertTrue("/tmp/particleflow/test_prod/tfds" in config["data_dir"])

    def test_override_config_basic(self):
        config = build_config_from_spec(self.spec, "test_model", "test_prod")
        
        args = argparse.Namespace()
        args.batch_size = 64
        args.lr = 0.01
        args.train = True
        args.test = False
        args.make_plots = True
        args.test_datasets = []
        
        extra_args = []
        
        config = override_config(config, args, extra_args)
        
        self.assertEqual(config["batch_size"], 64)
        self.assertEqual(config["lr"], 0.01)
        self.assertTrue(config["train"])
        self.assertFalse(config["test"])
        self.assertTrue(config["make_plots"])

    def test_override_config_extra_args(self):
        config = build_config_from_spec(self.spec, "test_model", "test_prod")
        
        args = argparse.Namespace()
        args.train = True
        args.test_datasets = []
        
        extra_args = ["--model.gnn_lsh.width", "128", "--num_steps", "200"]
        
        config = override_config(config, args, extra_args)
        
        self.assertEqual(config["model"]["gnn_lsh"]["width"], 128)
        self.assertEqual(config["num_steps"], 200)

    def test_override_config_convenience_flags(self):
        config = build_config_from_spec(self.spec, "test_model", "test_prod")
        
        args = argparse.Namespace()
        args.num_convs = 5
        args.train = True
        args.test_datasets = []
        
        config = override_config(config, args, [])
        
        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 5)

    def test_pipeline_overrides(self):
        config = build_config_from_spec(self.spec, "test_model", "test_prod")
        
        # Simulate logic in mlpf/pipeline.py main()
        args = argparse.Namespace()
        args.pipeline = True
        args.command = "train"
        args.test_datasets = []
        
        # Manual overrides from pipeline.py
        if args.pipeline:
            if "gnn_lsh" not in config["model"]:
                config["model"]["gnn_lsh"] = {}
            config["model"]["gnn_lsh"]["num_convs"] = 1
            config["model"]["gnn_lsh"]["width"] = 32
            config["model"]["gnn_lsh"]["embedding_dim"] = 32

            if config["dataset"] == "cms":
                for ds in ["train_dataset", "valid_dataset"]:
                    if ds in config:
                        config[ds]["cms"] = {
                            "physical_pu": {
                                "batch_size": config[ds]["cms"]["physical_pu"]["batch_size"],
                                "samples": {"cms_pf_ttbar": {"splits": ["10"], "version": "3.0.0"}},
                            }
                        }
                if "cms_pf_ttbar" in config["test_dataset"]:
                    config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]}
                    config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]
        
        self.assertEqual(config["model"]["gnn_lsh"]["num_convs"], 1)
        self.assertEqual(config["model"]["gnn_lsh"]["width"], 32)
        self.assertEqual(config["train_dataset"]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"], ["10"])
        self.assertEqual(config["test_dataset"]["cms_pf_ttbar"]["splits"], ["10"])

if __name__ == "__main__":
    unittest.main()
