import ast
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "BeingH"
    / "model"
    / "llm"
    / "qwen3_navit.py"
)


def load_route_function():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_effective_npu_attention_layout"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace["_effective_npu_attention_layout"]


class NpuAttentionLayoutRouteTest(unittest.TestCase):
    def test_bsnd_is_used_only_for_single_sample(self) -> None:
        route = load_route_function()
        self.assertEqual(route("BSND", 1), "BSND")
        self.assertEqual(route("BSND", 2), "BNSD")
        self.assertEqual(route("BSND", 8), "BNSD")

    def test_bnsd_is_never_changed(self) -> None:
        route = load_route_function()
        self.assertEqual(route("BNSD", 1), "BNSD")
        self.assertEqual(route("BNSD", 4), "BNSD")

    def test_hybrid_prefix_forces_bnsd_for_single_sample(self) -> None:
        route = load_route_function()
        self.assertEqual(route("BSND", 1, True), "BNSD")
        self.assertEqual(route("BSND", 8, True), "BNSD")


if __name__ == "__main__":
    unittest.main()
