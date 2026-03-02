import unittest
from mlpf.utils import resolve_path


class TestUtils(unittest.TestCase):
    def test_resolve_path_simple(self):
        spec = {"a": "val_a", "b": "val_b"}
        self.assertEqual(resolve_path("${a}", spec), "val_a")
        self.assertEqual(resolve_path("${b}", spec), "val_b")
        self.assertEqual(resolve_path("${a}/${b}", spec), "val_a/val_b")

    def test_resolve_path_nested_dict(self):
        spec = {"project": {"paths": {"storage_root": "/path/to/storage"}}}
        self.assertEqual(resolve_path("${project.paths.storage_root}", spec), "/path/to/storage")

    def test_resolve_path_recursive(self):
        spec = {
            "project": {"paths": {"storage_root": "/path/to/storage"}},
            "productions": {"cms": {"workspace_dir": "${project.paths.storage_root}/cms_workspace"}},
            "val_dir": "${productions.cms.workspace_dir}/val",
        }
        # One level of recursion
        self.assertEqual(resolve_path("${productions.cms.workspace_dir}", spec), "/path/to/storage/cms_workspace")
        # Two levels of recursion
        self.assertEqual(resolve_path("${val_dir}", spec), "/path/to/storage/cms_workspace/val")

    def test_resolve_path_unresolved(self):
        spec = {"a": "val_a"}
        # Should remain unchanged if key is not found
        self.assertEqual(resolve_path("${nonexistent}", spec), "${nonexistent}")
        self.assertEqual(resolve_path("${a}/${nonexistent}", spec), "val_a/${nonexistent}")


if __name__ == "__main__":
    unittest.main()
