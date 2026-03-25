import unittest
import yaml
import tempfile
import os
from mlpf.conf import MLPFConfig, Dataset, OptimizerType
from mlpf.pipeline import get_parser, Command


class TestPipelineConfig(unittest.TestCase):
    def setUp(self):
        # Create a mock spec file similar to particleflow_spec.yaml
        self.spec = {
            "project": {
                "workspace_dir": "/tmp/particleflow",
            },
            "models": {
                "defaults": {
                    "batch_size": 32,
                    "num_steps": 100,
                    "gpus": 1,
                    "optimizer": "adamw",
                },
                "test_model_gnn": {
                    "dataset": "cms",
                    "architecture": {
                        "type": "gnn_lsh",
                        "gnn_lsh": {
                            "num_convs": 2,
                            "width": 64,
                        },
                    },
                    "train_datasets": {"physical_pu": {"samples": [{"name": "cms_pf_ttbar", "version": "1.0.0", "splits": ["1"]}]}},
                    "validation_datasets": {"physical_pu": {"samples": [{"name": "cms_pf_ttbar", "version": "1.0.0", "splits": ["1"]}]}},
                    "test_datasets": [{"name": "cms_pf_ttbar", "version": "1.0.0"}],
                },
                "test_model_attention": {
                    "dataset": "clic",
                    "architecture": {
                        "type": "attention",
                        "attention": {
                            "num_convs": 2,
                            "num_heads": 4,
                        },
                    },
                    "train_datasets": {"physical": {"samples": [{"name": "clic_edm_ttbar_pf", "version": "1.0.0", "splits": ["1"]}]}},
                    "test_datasets": [{"name": "clic_edm_ttbar_pf", "version": "1.0.0"}],
                },
            },
            "productions": {
                "test_prod": {
                    "type": "cms",
                    "workspace_dir": "${project.workspace_dir}/test_prod",
                },
                "test_prod_clic": {
                    "type": "clic",
                    "workspace_dir": "${project.workspace_dir}/test_prod_clic",
                },
            },
        }
        self.temp_spec = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(self.spec, self.temp_spec)
        self.temp_spec_path = self.temp_spec.name
        self.temp_spec.close()

    def tearDown(self):
        if os.path.exists(self.temp_spec_path):
            os.unlink(self.temp_spec_path)

    def test_train_command_parsing(self):
        parser = get_parser()
        # Simulate: python mlpf/pipeline.py --spec-file spec.yaml --model-name test_model_gnn --production-name test_prod train --num-steps 500 --lr 0.005
        cmd_args = [
            "--spec-file",
            self.temp_spec_path,
            "--model-name",
            "test_model_gnn",
            "--production-name",
            "test_prod",
            "train",
            "--num-steps",
            "500",
            "--lr",
            "0.005",
        ]
        args, extra_args = parser.parse_known_args(cmd_args)

        # This part mimics main() in pipeline.py
        cmd = Command(args.command)
        self.assertEqual(cmd, Command.TRAIN)
        args.train = True
        args.test = True
        args.hpo = None
        args.ray_train = False

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        self.assertTrue(config_obj.train)
        self.assertTrue(config_obj.test)
        self.assertEqual(config_obj.num_steps, 500)
        self.assertEqual(config_obj.lr, 0.005)
        self.assertEqual(config_obj.dataset, Dataset.CMS)

    def test_test_command_parsing(self):
        parser = get_parser()
        cmd_args = ["--spec-file", self.temp_spec_path, "--model-name", "test_model_gnn", "--production-name", "test_prod", "test", "--gpus", "0"]
        args, extra_args = parser.parse_known_args(cmd_args)

        cmd = Command(args.command)
        self.assertEqual(cmd, Command.TEST)
        args.train = False
        args.test = True
        args.hpo = None
        args.ray_train = False

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        self.assertFalse(config_obj.train)
        self.assertTrue(config_obj.test)
        self.assertEqual(config_obj.gpus, 0)

    def test_ray_train_command_parsing(self):
        parser = get_parser()
        cmd_args = [
            "--spec-file",
            self.temp_spec_path,
            "--model-name",
            "test_model_gnn",
            "--production-name",
            "test_prod",
            "ray-train",
            "--ray-gpus",
            "2",
        ]
        args, extra_args = parser.parse_known_args(cmd_args)

        cmd = Command(args.command)
        self.assertEqual(cmd, Command.RAY_TRAIN)
        args.train = True
        args.test = True
        args.hpo = None
        args.ray_train = True
        args.gpus = args.ray_gpus

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        self.assertTrue(config_obj.train)
        self.assertEqual(config_obj.gpus, 2)

    def test_ray_hpo_command_parsing(self):
        parser = get_parser()
        cmd_args = [
            "--spec-file",
            self.temp_spec_path,
            "--model-name",
            "test_model_gnn",
            "--production-name",
            "test_prod",
            "ray-hpo",
            "--name",
            "my_hpo_experiment",
            "--ray-gpus",
            "1",
        ]
        args, extra_args = parser.parse_known_args(cmd_args)

        cmd = Command(args.command)
        self.assertEqual(cmd, Command.RAY_HPO)
        args.train = True
        args.test = False
        args.hpo = args.name
        args.ray_train = False
        args.gpus = args.ray_gpus

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        self.assertTrue(config_obj.train)
        self.assertFalse(config_obj.test)
        self.assertEqual(config_obj.gpus, 1)

    def test_extra_args_dot_notation(self):
        parser = get_parser()
        cmd_args = [
            "--spec-file",
            self.temp_spec_path,
            "--model-name",
            "test_model_gnn",
            "--production-name",
            "test_prod",
            "train",
            "--model.gnn_lsh.width=128",
            "--optimizer=sgd",
        ]
        args, extra_args = parser.parse_known_args(cmd_args)

        # In actual pipeline.py, extra_args would contain things like ['--model.gnn_lsh.width=128', '--optimizer=sgd']
        # if they are not defined in the parser.
        # But get_parser() might not catch them if they are not in the known args.

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        self.assertEqual(config_obj.model.gnn_lsh.width, 128)
        self.assertEqual(config_obj.optimizer, OptimizerType.SGD)

    def test_pipeline_flag_ci_cd(self):
        parser = get_parser()
        cmd_args = [
            "--spec-file",
            self.temp_spec_path,
            "--model-name",
            "test_model_attention",
            "--production-name",
            "test_prod_clic",
            "--pipeline",
            "train",
        ]
        args, extra_args = parser.parse_known_args(cmd_args)

        args.train = True
        args.test = True
        args.hpo = None
        args.ray_train = False

        config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)

        # Check pipeline overrides for clic
        self.assertEqual(config_obj.model.attention.num_convs, 1)
        self.assertEqual(config_obj.model.attention.num_heads, 2)
        self.assertEqual(config_obj.model.attention.head_dim, 2)

        # Check dataset override for clic
        self.assertIn("clic_edm_ttbar_pf", config_obj.train_dataset["clic"]["physical"].samples)
        self.assertEqual(config_obj.train_dataset["clic"]["physical"].samples["clic_edm_ttbar_pf"].splits, ["10"])

    def test_invalid_model_name(self):
        parser = get_parser()
        cmd_args = ["--spec-file", self.temp_spec_path, "--model-name", "non_existent_model", "--production-name", "test_prod", "train"]
        args, extra_args = parser.parse_known_args(cmd_args)

        with self.assertRaisesRegex(ValueError, "Model non_existent_model not found in spec"):
            MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)


if __name__ == "__main__":
    unittest.main()
