import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_SCRIPT = ROOT / "examples" / "quickstart_env.py"


def test_quickstart_script_exists():
    assert QUICKSTART_SCRIPT.exists()


def test_quickstart_script_runs_when_gymnasium_is_installed():
    pytest.importorskip("gymnasium")

    runpy.run_path(str(QUICKSTART_SCRIPT), run_name="__main__")
