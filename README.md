# RL-ADN

RL-ADN is a Python library for deep reinforcement learning research on energy storage dispatch in active distribution networks. It packages network data, environment logic, baseline optimization code, and fast Laurent power-flow utilities used in the accompanying research line.

## Quickstart

Install runtime dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

Install the development toolchain:

```bash
py -3 -m pip install -r requirements-dev.txt
```

Run the lightweight test suite:

```bash
py -3 -m pytest tests -q -m "not powerflow"
```

Some power-flow validation tests require `pandapower`. If it is not installed, those tests are skipped automatically.

## First Import

```python
from rl_adn import PowerNetEnv, make_env_config

config = make_env_config()
env = PowerNetEnv(config)
state = env.reset()
```

Run the script-style quickstart:

```bash
py -3 examples/quickstart_env.py
```

## Repository Structure

- `rl_adn/`: package source code
- `tests/`: smoke and domain validation tests
- `examples/`: script-first quickstart plus notebooks
- `docs/`: Sphinx documentation sources

## Highlights

- Flexible active distribution network environment modeling
- Laurent power flow solver for faster training-time simulation
- DRL algorithms and optimization baselines in the same repository
- Bundled network and time-series datasets for reproducible experiments

## Background

The library was originally released alongside the RL-ADN research paper on optimal battery dispatch in distribution networks. The `codex/develop` branch is being used to harden the repository into a cleaner long-lived development branch for future extensions.

## Recommended Learning Path

1. Run `examples/quickstart_env.py` for the minimal package-backed environment flow.
2. Open `examples/Customize_env.ipynb` to understand configuration customization.
3. Open the DDPG training notebook once the environment baseline is clear.

## More Detail

For tutorial-style material, see the project [wiki](https://github.com/EnergyQuantResearch/RL-ADN/wiki).
