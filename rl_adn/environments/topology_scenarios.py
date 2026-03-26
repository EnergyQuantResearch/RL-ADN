from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


Edge = Tuple[int, int]
Rewire = Tuple[Edge, Edge]


@dataclass(frozen=True)
class TopologyScenario:
    feeder_id: str
    node_count: int
    scenario_id: str
    description: str
    rewires: Tuple[Rewire, ...]


def _scenario(feeder_id: str, node_count: int, scenario_id: str, description: str, rewires: Sequence[Rewire]):
    return TopologyScenario(
        feeder_id=feeder_id,
        node_count=node_count,
        scenario_id=scenario_id,
        description=description,
        rewires=tuple(rewires),
    )


_SCENARIOS: Dict[int, Dict[str, TopologyScenario]] = {
    34: {
        "TP1": _scenario("34-bus", 34, "TP1", "Baseline topology", ()),
        "TP2": _scenario("34-bus", 34, "TP2", "One within-branch reconnection", [((25, 26), (24, 26))]),
        "TP3": _scenario("34-bus", 34, "TP3", "One within-branch reconnection", [((32, 33), (31, 33))]),
        "TP4": _scenario(
            "34-bus",
            34,
            "TP4",
            "Two within-branch reconnections",
            [((11, 12), (10, 12)), ((29, 30), (28, 30))],
        ),
        "TP5": _scenario(
            "34-bus",
            34,
            "TP5",
            "Two within-branch reconnections",
            [((15, 16), (14, 16)), ((33, 34), (32, 34))],
        ),
        "TP6": _scenario("34-bus", 34, "TP6", "Whole-branch reconnection", [((10, 31), (8, 31))]),
        "TP7": _scenario("34-bus", 34, "TP7", "Whole-branch reconnection", [((10, 11), (9, 11))]),
    },
    69: {
        "TP1": _scenario("69-bus", 69, "TP1", "Baseline topology", ()),
        "TP2": _scenario("69-bus", 69, "TP2", "One within-branch reconnection", [((67, 68), (13, 68))]),
        "TP3": _scenario("69-bus", 69, "TP3", "One within-branch reconnection", [((45, 46), (44, 46))]),
        "TP4": _scenario(
            "69-bus",
            69,
            "TP4",
            "Two within-branch reconnections",
            [((35, 36), (34, 36)), ((13, 69), (14, 69))],
        ),
        "TP5": _scenario(
            "69-bus",
            69,
            "TP5",
            "Two within-branch reconnections",
            [((50, 51), (49, 51)), ((52, 53), (9, 53))],
        ),
        "TP6": _scenario("69-bus", 69, "TP6", "Whole-branch reconnection", [((12, 13), (11, 13))]),
        "TP7": _scenario("69-bus", 69, "TP7", "Whole-branch reconnection", [((10, 54), (8, 54))]),
    },
}


def list_topology_scenario_ids(node_count: int) -> List[str]:
    if node_count not in _SCENARIOS:
        raise KeyError(f"No topology scenarios are registered for feeder {node_count}")
    return list(_SCENARIOS[node_count].keys())


def get_topology_scenario(node_count: int, scenario_id: str) -> TopologyScenario:
    if node_count not in _SCENARIOS:
        raise KeyError(f"No topology scenarios are registered for feeder {node_count}")
    try:
        return _SCENARIOS[node_count][scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown topology scenario '{scenario_id}' for feeder {node_count}") from exc


def get_topology_scenarios(node_count: int) -> Dict[str, TopologyScenario]:
    if node_count not in _SCENARIOS:
        raise KeyError(f"No topology scenarios are registered for feeder {node_count}")
    return dict(_SCENARIOS[node_count])
