from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_algorithm_utility_facade_exports_expected_symbols():
    from rl_adn.algorithms import utility

    assert hasattr(utility, "Config")
    assert set(utility.__all__) == {"Config", "ReplayBuffer", "SumTree", "build_mlp", "get_episode_return", "get_optim_param"}

    if importlib.util.find_spec("torch") is not None:
        for name in ("ReplayBuffer", "SumTree", "build_mlp", "get_episode_return", "get_optim_param"):
            assert hasattr(utility, name)


def test_benchmark_module_imports_without_pyomo_side_effects():
    module = importlib.import_module("rl_adn.benchmarks.pyomo_timeseries_pandapower")
    assert hasattr(module, "DispatchBenchmarkData")
    assert hasattr(module, "construct_opf_model")


def test_benchmark_frame_conversion_orders_rows_and_columns():
    from rl_adn.benchmarks.pyomo_timeseries_pandapower import convert_indexed_values_to_frame

    frame = convert_indexed_values_to_frame({(1, 2): 5, (0, 1): 3, (0, 2): 4})
    assert list(frame.index) == [0, 1]
    assert list(frame.columns) == [1, 2]
    assert frame.loc[0, 1] == 3


def test_active_power_data_manager_extracts_day_node_matrix(tmp_path: Path):
    from rl_adn.data_augment.data_augment import ActivePowerDataManager

    periods = 96
    frame = pd.DataFrame(
        {
            "date_time": pd.date_range("2021-01-01", periods=periods, freq="15min", tz="UTC"),
            "active_power_node_1": range(periods),
            "active_power_node_2": range(periods, periods * 2),
        }
    )
    csv_path = tmp_path / "active_power.csv"
    frame.to_csv(csv_path, index=False)

    manager = ActivePowerDataManager(str(csv_path))
    active_power_data = manager.get_active_power_data()

    assert active_power_data.shape == (2, periods)


def test_construct_opf_model_requires_pyomo_when_dependency_missing():
    from rl_adn.benchmarks.pyomo_timeseries_pandapower import BatterySpec, DispatchBenchmarkData, construct_opf_model

    data = DispatchBenchmarkData(
        times=(0,),
        nodes=(0, 1),
        lines=((0, 1),),
        tb={0: 1, 1: 0},
        pd=[[0.0, 0.0]],
        qd=[[0.0, 0.0]],
        r={(0, 1): 0.1},
        x={(0, 1): 0.1},
        battery_nodes=frozenset({1}),
        price=[1.0],
        battery=BatterySpec(),
    )

    if importlib.util.find_spec("pyomo") is None:
        with pytest.raises(ImportError):
            construct_opf_model(1.0, 0.95, 1.05, data)
    else:
        model = construct_opf_model(1.0, 0.95, 1.05, data)
        assert model is not None
