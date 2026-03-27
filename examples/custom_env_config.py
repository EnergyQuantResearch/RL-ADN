from __future__ import annotations

from dataclasses import replace

from rl_adn import PowerNetEnv, make_env_config


def main() -> None:
    base_config = make_env_config(node=34, topology_scenario="TP3")
    custom_config = replace(
        base_config,
        battery_nodes=(11, 15, 26),
        battery=replace(base_config.battery, max_charge_kw=25.0, max_discharge_kw=25.0),
    )

    env = PowerNetEnv(custom_config)
    state, info = env.reset(seed=2026)

    print("Battery nodes:", custom_config.battery_nodes)
    print("Topology scenario:", info["topology_scenario"])
    print("State shape:", state.shape)


if __name__ == "__main__":
    main()
