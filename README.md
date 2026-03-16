# RL-ADN

RL-ADN is a Python library for deep reinforcement learning research on energy storage dispatch in active distribution networks. It packages network data, environment logic, baseline optimization code, and fast Laurent power-flow utilities used in the accompanying research line.

## Quickstart

Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

Run the test suite:

```bash
py -3 -m pytest tests -q
```

Some power-flow validation tests require `pandapower`. If it is not installed, those tests are skipped automatically.

## Repository Structure

- `rl_adn/`: package source code
- `tests/`: smoke and domain validation tests
- `examples/`: notebooks and example workflows
- `docs/`: Sphinx documentation sources

## Highlights

- Flexible active distribution network environment modeling
- Laurent power flow solver for faster training-time simulation
- DRL algorithms and optimization baselines in the same repository
- Bundled network and time-series datasets for reproducible experiments

## Background

The library was originally released alongside the RL-ADN research paper on optimal battery dispatch in distribution networks. The `codex/develop` branch is being used to harden the repository into a cleaner long-lived development branch for future extensions.

## More Detail

For tutorial-style material, see the project [wiki](https://github.com/ShengrenHou/RL-ADN/wiki).
