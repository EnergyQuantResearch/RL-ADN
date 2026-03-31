import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SCRIPTS = [
    ROOT / "examples" / "quickstart_env.py",
    ROOT / "examples" / "custom_env_config.py",
    ROOT / "examples" / "live_dashboard_rollout.py",
    ROOT / "examples" / "topology_scenarios.py",
    ROOT / "examples" / "training_smoke.py",
]


def test_example_scripts_exist():
    for script in EXAMPLE_SCRIPTS:
        assert script.exists()


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS)
def test_example_script_runs_when_gymnasium_is_installed(script_path: Path):
    pytest.importorskip("gymnasium")
    if script_path.name == "training_smoke.py":
        pytest.importorskip("torch")
    if script_path.name == "live_dashboard_rollout.py":
        pytest.skip("live dashboard script is covered by dedicated CLI smoke tests")

    runpy.run_path(str(script_path), run_name="__main__")
