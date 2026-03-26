import os
import warnings

import numpy as np
import pandas as pd

from rl_adn.data import GeneralPowerDataManager


def test_GeneralPowerDataManager(tmp_path):
    # Generate sample data
    sample_data = {
        "date_time": pd.date_range(start="2021-01-01", periods=24 * 30, freq="h", tz="UTC"),
        "active_power_node_1": np.random.rand(24 * 30),
        "active_power_node_2": np.random.rand(24 * 30),
        "reactive_power_node_1": np.random.rand(24 * 30),
        "price_node_1": np.random.rand(24 * 30),
    }
    df = pd.DataFrame(sample_data)
    datapath = tmp_path / "sample_data.csv"
    df.to_csv(datapath, index=False)

    # Test initialization
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        data = GeneralPowerDataManager(datapath)

    future_warnings = [w for w in caught_warnings if issubclass(w.category, FutureWarning)]
    assert not future_warnings, "GeneralPowerDataManager emitted FutureWarning during initialization"
    assert isinstance(data.df, pd.DataFrame), "DataFrame not initialized"
    assert len(data.active_power_cols) == 2, "Active power columns not detected correctly"
    assert len(data.reactive_power_cols) == 1, "Reactive power columns not detected correctly"
    assert len(data.price_col) == 1, "Price columns not detected correctly"

    # Test select_timeslot_data
    timeslot_data = data.select_timeslot_data(2021, 1, 1, 0)
    assert len(timeslot_data) == 4, "Timeslot data not fetched correctly"

    # Test select_day_data
    day_data = data.select_day_data(2021, 1, 1)
    assert day_data.shape[0] == 24, "Day data not fetched correctly"

    # Test list_dates
    dates = data.list_dates()
    assert len(dates) == 30, "List dates not working correctly"

    # Test random_date
    date = data.random_date()
    assert isinstance(date, tuple) and len(date) == 3, "Random date function not working correctly"

    # Test split_data_set
    train_dates = data.train_dates
    test_dates = data.test_dates
    assert len(train_dates) == 22, "Training dates not split correctly"
    assert len(test_dates) == 8, "Testing dates not split correctly"

    # Cleanup
    os.remove(datapath)
