"""Verify GOATBench eval skips crashed scenes and still produces output."""
import ast
import os
import tempfile


# =============================================================================
# Helper: find a method node inside an AST class node.
# =============================================================================
def _find_method(tree: ast.AST, class_name: str, method_name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


# =============================================================================
# Tests
# =============================================================================


def test_save_results_writes_exactly_9_files():
    """save_results must open 6 .pkl + 3 .json files for writing."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "logger_goatbench.py"
    )
    with open(source_path) as f:
        tree = ast.parse(f.read())

    method = _find_method(tree, "Logger", "save_results")
    assert method is not None, "save_results not found in Logger class"

    # Collect all complete filename patterns from open(...) calls.
    filenames = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                arg = node.args[0]
                if isinstance(arg, ast.Call):
                    if len(arg.args) >= 2:
                        fname_arg = arg.args[1]
                        if isinstance(fname_arg, ast.JoinedStr):
                            parts = []
                            for val in fname_arg.values:
                                if isinstance(val, ast.Constant):
                                    parts.append(val.value)
                                elif isinstance(val, ast.FormattedValue):
                                    parts.append("{...}")
                            filenames.append("".join(parts))
                        elif isinstance(fname_arg, ast.Constant):
                            filenames.append(fname_arg.value)

    pkl_names = [n for n in filenames if n.endswith(".pkl")]
    json_names = [n for n in filenames if n.endswith(".json")]

    assert len(pkl_names) == 6, f"expected 6 pkl filenames, got {pkl_names}"
    assert len(json_names) == 3, f"expected 3 json filenames, got {json_names}"

    # Names should cover the expected categories
    expected_patterns = [
        "success_by_snapshot",
        "success_by_distance",
        "spl_by_snapshot",
        "spl_by_distance",
        "success_by_task",
        "spl_by_task",
    ]
    for pat in expected_patterns:
        assert any(pat in n for n in pkl_names), f"missing pkl: {pat}"

    json_patterns = [
        "n_filtered_snapshots",
        "n_total_snapshots",
        "n_total_frames",
    ]
    for pat in json_patterns:
        assert any(pat in n for n in json_names), f"missing json: {pat}"


def test_corrupted_scene_set_exists():
    """CORRUPTED_SCENES must be defined as a set in run_goatbench_evaluation.py."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "run_goatbench_evaluation.py"
    )
    with open(source_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) or isinstance(node, ast.Assign):
            for target in (node.targets if hasattr(node, "targets") else [node.target]):
                if isinstance(target, ast.Name) and target.id == "CORRUPTED_SCENES":
                    value = node.value
                    assert isinstance(value, (ast.Set, ast.Call)), (
                        f"CORRUPTED_SCENES is not a set literal or set() call, "
                        f"got {ast.dump(value)}"
                    )
                    return
    raise AssertionError("CORRUPTED_SCENES not found in run_goatbench_evaluation.py")


def test_corrupted_scenes_helpers_exist():
    """Helper functions for corrupted scene persistence must exist."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "run_goatbench_evaluation.py"
    )
    with open(source_path) as f:
        tree = ast.parse(f.read())

    expected_helpers = {
        "_corrupted_scenes_path",
        "_save_corrupted_scenes",
        "_load_corrupted_scenes",
    }
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in expected_helpers:
            found.add(node.name)
    missing = expected_helpers - found
    assert not missing, f"Missing helper functions: {missing}"
