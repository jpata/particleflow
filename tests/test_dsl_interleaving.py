import sys
import os
import unittest

# Add the project root to sys.path to ensure mlpf is importable
sys.path.append(os.getcwd())

from mlpf.standalone.dsl import (
    parse_dsl,
    config_to_string,
    ModelConfig,
    HEPTConfig,
    GlobalConfig,
    StandardConfig,
)
from mlpf.standalone.run_evolution import generate_random_config, mutate_config


class TestDSLInterleaving(unittest.TestCase):
    def test_interleaved_parsing(self):
        """Test that interleaved DSL strings are correctly parsed and preserved."""
        # A DSL with HEPT followed by Global followed by HEPT
        # Note: *1 is omitted by config_to_string
        dsl = "i(55,128,256,default)|h(16,128,512)*2+g(16,128,512)+h(16,128,512)*2|o(8,256,default)"
        cfg = parse_dsl(dsl)

        shared_layers = cfg.backbone["shared"]
        self.assertEqual(len(shared_layers), 5)
        self.assertIsInstance(shared_layers[0], HEPTConfig)
        self.assertIsInstance(shared_layers[1], HEPTConfig)
        self.assertIsInstance(shared_layers[2], GlobalConfig)
        self.assertIsInstance(shared_layers[3], HEPTConfig)
        self.assertIsInstance(shared_layers[4], HEPTConfig)

        # Check round-trip
        round_trip = config_to_string(cfg)
        self.assertEqual(round_trip, dsl)

    def test_random_generation_interleaving(self):
        """Verify that random generation actually produces interleaved layers."""
        interleaved_found = False
        for _ in range(50):
            dsl = generate_random_config()
            cfg = parse_dsl(dsl)

            # Check if any branch has mixed types
            for branch_layers in cfg.backbone.values():
                if len(branch_layers) > 1:
                    types = [type(_l) for _l in branch_layers]
                    if len(set(types)) > 1:
                        interleaved_found = True
                        break
            if interleaved_found:
                break

        self.assertTrue(interleaved_found, "Random generation should produce interleaved layer types within 50 attempts.")

    def test_mutation_preserves_validity(self):
        """Verify that mutating an interleaved config results in a valid config."""
        dsl = "i(55,128,256,default)|h(16,128,512)*2+g(16,128,512)*1|o(8,256,default)"
        cfg = parse_dsl(dsl)

        for _ in range(20):
            cfg = mutate_config(cfg)
            new_dsl = config_to_string(cfg)
            try:
                parsed_cfg = parse_dsl(new_dsl)
                self.assertIsInstance(parsed_cfg, ModelConfig)
            except Exception as e:
                self.fail(f"Mutation resulted in invalid DSL '{new_dsl}': {e}")

    def test_layer_type_mutation(self):
        """Verify that individual layers can change type during mutation."""
        layer = StandardConfig(num_heads=8, embedding_dim=128, width=512)

        # We might need multiple attempts to see a type change due to randomness
        type_changed = False
        from mlpf.standalone.run_evolution import mutate_layer

        for _ in range(100):
            mutated = mutate_layer(layer)
            if not isinstance(mutated, StandardConfig):
                type_changed = True
                break

        self.assertTrue(type_changed, "Layer mutation should eventually change the layer type.")


if __name__ == "__main__":
    unittest.main()
