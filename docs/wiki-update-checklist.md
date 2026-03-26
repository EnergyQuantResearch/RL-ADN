# Wiki Update Checklist

This checklist tracks the GitHub Wiki updates needed after Phase A topology scenario support landed in RL-ADN.

## New or Updated Wiki Pages

- `Home`
  - add a short Phase A summary
  - link to topology scenarios, graph exports, and quickstart examples
- `Getting Started`
  - update the first import example to use `reset(return_info=True)`
  - mention `requirements-dev.txt` and lightweight test command
- `Environment Configuration`
  - document `make_env_config(..., topology_mode, topology_scenario, topology_pool, return_graph)`
  - explain fixed vs scenario-pool behavior
- `Topology Scenarios`
  - add `TP1-TP7` tables for `34-bus` and `69-bus`
  - explain the rule for new edges: inherit `R/X/B/TAP` from the replaced edge
  - state clearly that topology changes only on `reset()`
- `Graph Interface`
  - document `get_topology_metadata()`
  - document `get_graph_data()`
  - explain that outputs are plain Python/NumPy structures
- `Examples`
  - point to `examples/quickstart_env.py`
  - add one fixed-scenario example and one scenario-pool example
- `Known Limits`
  - note that Phase A does not add topology control actions
  - note that `69-bus` currently needs a user-provided time-series CSV for rollout

## Content Details To Include

- Explain that Phase A follows the paper-style setup: topology as scenario variation, not as an agent action.
- Clarify that the default flat RL state stays unchanged.
- Clarify that topology-aware graph outputs are additive interfaces for future GNN work.
- Mention that all packaged topology scenarios are validated to remain connected and radial.

## Verification Before Publishing Wiki Changes

- Confirm examples in Wiki match the current public API.
- Confirm scenario ids are exactly `TP1` through `TP7`.
- Confirm `34-bus` and `69-bus` feeder names match the code and README wording.
- Confirm the `69-bus` data limitation is still accurate at publish time.
