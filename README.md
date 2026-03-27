# RL-ADN

RL-ADN is a Python library for deep reinforcement learning research on energy storage dispatch in active distribution networks. It packages network data, environment logic, baseline optimization code, and fast Laurent power-flow utilities used in the accompanying research line.

## Phase A Status

RL-ADN now supports `topology-as-scenario` for the `34-bus` and `69-bus` feeders. In this phase, topology does not become part of the RL action space. Instead, the environment can switch among hand-authored radial topology scenarios at `reset()` time, following the `TP1–TP7` evaluation style used in the topology-aware GNN transferability paper.

## Quickstart

Install the package:

```bash
py -3 -m pip install .
```

Install the development toolchain:

```bash
py -3 -m pip install -e .[dev]
```

Run the lightweight verification suite:

```bash
py -3 -m pytest -q -m "not powerflow"
```

Power-flow validation tests require the optional `pandapower` extra and are skipped when it is not installed.

## First Import

```python
from rl_adn import PowerNetEnv, make_env_config

config = make_env_config()
env = PowerNetEnv(config)
state, info = env.reset(seed=2026)
```

Run the script-style quickstart:

```bash
py -3 examples/quickstart_env.py
```

## Topology Scenarios

Phase A adds curated topology scenario pools for the `34-bus` and `69-bus` feeders:

- `TP1`: baseline topology
- `TP2–TP7`: reconfigured radial topologies derived from the paper's in-network reconnection cases

Use a fixed named scenario:

```python
from rl_adn import PowerNetEnv, make_env_config

config = make_env_config(node=34, topology_scenario="TP4", return_graph=True)
env = PowerNetEnv(config)
state, info = env.reset(seed=2026)
print(info["topology_scenario"])
```

Sample from a scenario pool at reset time:

```python
config = make_env_config(
    node=34,
    topology_mode="scenario_pool",
    topology_pool=["TP2", "TP3", "TP4"],
    return_graph=True,
)
env = PowerNetEnv(config)
state, info = env.reset(seed=2026)
```

Inspect the active topology for later GNN work:

```python
metadata = env.get_topology_metadata()
graph = env.get_graph_data()
```

`metadata` includes feeder id, scenario id, node count, edge count, and active edges. `graph` returns plain NumPy/Python structures such as adjacency and edge index.

## Public API

The stable package surface is:

- `Battery`
- `BatteryConfig`
- `EnvConfig`
- `TopologyConfig`
- `GeneralPowerDataManager`
- `PowerNetEnv`
- `make_env_config(...)`

`make_env_config(...)` now returns a typed `EnvConfig` dataclass rather than a loose dictionary.

`PowerNetEnv` follows Gymnasium semantics:

```python
obs, info = env.reset(seed=2026)
next_obs, reward, terminated, truncated, info = env.step(action)
```

## Supported Examples

The supported example entrypoints are the Python scripts described in `examples/README.md`.

The notebooks are retained only as supplementary reference material and are no longer treated as the primary supported workflow.

## Repository Structure

- `rl_adn/`: package source code
- `tests/`: smoke and domain validation tests
- `examples/`: supported scripts plus supplementary notebooks
- `docs/`: Sphinx documentation sources

## Highlights

- Gymnasium-style active distribution network environment
- Laurent power flow solver for faster training-time simulation
- DRL algorithms and optimization baselines in the same repository
- Bundled network and time-series datasets for reproducible experiments
- Reset-time topology scenario support for 34-bus and 69-bus feeders
- Graph/topology observation exports for future GNN-based controllers

## Background

The library was originally released alongside the RL-ADN research paper on optimal battery dispatch in distribution networks. The `codex/develop` branch is being used to harden the repository into a cleaner long-lived development branch for future extensions.

## Recommended Learning Path

1. Run `examples/quickstart_env.py` for the minimal package-backed environment flow.
2. Read the typed config surface through `rl_adn.make_env_config(...)`.
3. Try fixed and pooled topology scenarios before moving on to GNN-based experiments.
4. Treat notebooks as archival/supplementary material after the script workflow is clear.

## Current Limits

- Phase A only supports topology variation as an environment scenario, not as an RL control action.
- `34-bus` packaged time-series data is available out of the box.
- `69-bus` topology scenarios are supported, but environment rollout currently requires a user-provided time-series CSV because a packaged `69_node_time_series.csv` is not yet included.

## More Detail

For tutorial-style material, see the project [wiki](https://github.com/EnergyQuantResearch/RL-ADN/wiki).
