from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SPEC = spec_from_file_location("acquire_commons", Path("dataset-builder/acquire_commons.py"))
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_allows_permissive_commercial_licenses():
    for value in ["CC0", "CC0 1.0", "Public domain", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0"]:
        assert MODULE.is_allowed(value), value


def test_rejects_ambiguous_or_restrictive_licenses():
    for value in ["", "Unknown", "CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-SA 4.0", "All rights reserved"]:
        assert not MODULE.is_allowed(value), value
