from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location

SPEC = spec_from_file_location("build_dataset", Path("dataset-builder/build_dataset.py"))
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_group_assignment_is_deterministic_and_exclusive():
    groups = [f"video-{i}" for i in range(20)]
    first = MODULE.assign_groups(groups, seed=7)
    second = MODULE.assign_groups(groups, seed=7)
    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == {"train", "val", "test"}


def test_same_source_never_crosses_splits():
    groups = ["phone-a"] * 10 + ["phone-b"] * 10 + ["phone-c"] * 10
    mapping = MODULE.assign_groups(groups, seed=1)
    assert len(mapping) == 3
    assert all(value in {"train", "val", "test"} for value in mapping.values())
