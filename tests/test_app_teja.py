# ============================================================
# tests/test_app_teja.py
# ============================================================
# WHY: Smoke test my Phase 4 copy of the example module.
#
# Run:
#   uv run python -m pytest

from mlstudio import app_teja


def test_app_teja_has_main() -> None:
    """Verify my copy of the module exposes a main function."""
    assert callable(app_teja.main)
