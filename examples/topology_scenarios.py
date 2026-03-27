from __future__ import annotations

from rl_adn import PowerNetEnv, make_env_config


def main() -> None:
    config = make_env_config(
        node=34,
        topology_mode="scenario_pool",
        topology_pool=["TP2", "TP3", "TP4"],
        return_graph=True,
    )
    env = PowerNetEnv(config)
    _state, info = env.reset(seed=2026)
    metadata = env.get_topology_metadata()
    graph = env.get_graph_data()

    print("Sampled scenario:", info["topology_scenario"])
    print("Active edge count:", metadata["edge_count"])
    print("Adjacency shape:", graph["adjacency"].shape)


if __name__ == "__main__":
    main()
