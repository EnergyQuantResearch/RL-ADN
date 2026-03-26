"""Time-series data loading and preprocessing for RL-ADN."""

import random
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


class GeneralPowerDataManager:
    """
    A class to manage and preprocess time series data for power systems.

    Attributes:
        df (pd.DataFrame): The original data.
        data_array (np.ndarray): Array representation of the data.
        active_power_cols (List[str]): List of columns related to active power.
        reactive_power_cols (List[str]): List of columns related to reactive power.
        renewable_active_power_cols (List[str]): List of columns related to renewable active power.
        renewable_reactive_power_cols (List[str]): List of columns related to renewable reactive power.
        price_col (List[str]): List of columns related to price.
        train_dates (List[Tuple[int, int, int]]): List of training dates.
        test_dates (List[Tuple[int, int, int]]): List of testing dates.
        time_interval (int): Time interval of the data in minutes.
    """

    def __init__(self, datapath: str) -> None:
        """
        Initialize the GeneralPowerDataManager object.

        Parameters:
            datapath (str): Path to the CSV file containing the data.
        """
        if datapath is None:
            raise ValueError("Please input the correct datapath")

        data = pd.read_csv(datapath)
        if data.empty:
            raise ValueError("The provided dataset is empty and cannot be processed")

        # Prepare a datetime index and keep the DataFrame sorted for predictable slicing.
        data = self._prepare_datetime_index(data)

        min_date = data.index.min()
        max_date = data.index.max()
        self.time_interval = self._infer_time_interval(data.index)
        self.date_range = (min_date, max_date)

        # Initialize other attributes
        self.df = data
        self.data_array = data.values

        # Identify columns matching expected patterns for later use.
        self.active_power_cols = self._get_columns_matching(r"active_power(_\w+)?")
        self.reactive_power_cols = self._get_columns_matching(r"reactive_power(_\w+)?")
        self.renewable_active_power_cols = self._get_columns_matching(r"renewable_active_power(_\w+)?")
        self.renewable_reactive_power_cols = self._get_columns_matching(r"renewable_reactive_power(_\w+)?")
        self.price_col = self._get_columns_matching(r"price(_\w+)?")
        # Calculate max and min for each type of power
        self.active_power_max = self.df[self.active_power_cols].max().max()
        self.active_power_min = self.df[self.active_power_cols].min().min()

        self.reactive_power_max = self.df[self.reactive_power_cols].max().max() if self.reactive_power_cols else None
        self.reactive_power_min = self.df[self.reactive_power_cols].min().min() if self.reactive_power_cols else None

        self.renewable_active_power_max = self.df[self.renewable_active_power_cols].max().max() if self.renewable_active_power_cols else None
        self.renewable_active_power_min = self.df[self.renewable_active_power_cols].min().min() if self.renewable_active_power_cols else None

        self.renewable_reactive_power_max = self.df[self.renewable_reactive_power_cols].max().max() if self.renewable_reactive_power_cols else None
        self.renewable_reactive_power_min = self.df[self.renewable_reactive_power_cols].min().min() if self.renewable_reactive_power_cols else None
        self.price_min = self.df[self.price_col].min().values[0] if self.price_col else None
        self.price_max = self.df[self.price_col].max().values[0] if self.price_col else None
        # split the train and test dates
        self.train_dates = []
        self.test_dates = []
        self.split_data_set()
        self._replace_nan()
        self._check_for_nan()

    def _prepare_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate the datetime index used across the manager.

        This keeps the index conversion and sorting in a single place so other
        methods can assume a consistent, timezone-aware index.
        """
        index_column = "date_time" if "date_time" in data.columns else data.columns[0]

        # Copy to avoid mutating the caller's DataFrame unexpectedly.
        data = data.copy()
        data.set_index(index_column, inplace=True)

        try:
            data.index = pd.to_datetime(data.index)
        except (ValueError, TypeError) as exc:
            raise ValueError("Failed to parse the index into datetime values") from exc

        data.sort_index(inplace=True)
        return data

    def _infer_time_interval(self, index: pd.DatetimeIndex) -> int:
        """Infer the time interval in minutes from the datetime index."""
        if len(index) < 2:
            raise ValueError("At least two records are required to infer the time interval")

        # Use the mode of the time differences to stay robust to occasional gaps.
        deltas = index.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty or (deltas <= 0).any():
            raise ValueError("Invalid or non-increasing timestamps detected in the dataset")

        interval_minutes = int(deltas.mode().iloc[0] / 60)
        if interval_minutes == 0:
            raise ValueError("Computed time interval is zero minutes, please verify the input data")

        return interval_minutes

    def _get_columns_matching(self, pattern: str) -> List[str]:
        """Return column names that match the provided regex pattern."""
        return [col for col in self.df.columns if re.fullmatch(pattern, col)]

    def _replace_nan(self) -> None:
        """
        Replace NaN values in the data with interpolated values or the average of the surrounding values.
        """
        self.df.interpolate(inplace=True)
        self.df.bfill(inplace=True)
        self.df.ffill(inplace=True)

    def _check_for_nan(self) -> None:
        """
        Check if any of the arrays contain NaN values and raise an error if they do
        """
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("Data still contains NaN values after preprocessing")

    def select_timeslot_data(self, year: int, month: int, day: int, timeslot: int) -> np.ndarray:
        """
        Select data for a specific timeslot on a specific day.

        Parameters:
            year (int): The year of the date.
            month (int): The month of the date.
            day (int): The day of the date.
            timeslot (int): The timeslot index.

        Returns:
            np.ndarray: The data for the specified timeslot.
        """
        dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz="UTC") + pd.Timedelta(minutes=self.time_interval * timeslot)
        row = self.df.loc[dt]
        return row.values

    def select_day_data(self, year: int, month: int, day: int) -> np.ndarray:
        """
        Select data for a specific day.

        Parameters:
            year (int): The year of the date.
            month (int): The month of the date.
            day (int): The day of the date.

        Returns:
            np.ndarray: The data for the specified day.
        """
        start_dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz="UTC")
        end_dt = start_dt + pd.Timedelta(days=1)
        day_data = self.df.loc[start_dt : end_dt - pd.Timedelta(minutes=1), :]
        return day_data.values

    def list_dates(self) -> List[Tuple[int, int, int]]:
        """
        List all available dates in the data.

        Returns:
            List[Tuple[int, int, int]]: A list of available dates as (year, month, day).
        """
        normalized_dates = self.df.index.normalize().unique()
        return [(ts.year, ts.month, ts.day) for ts in normalized_dates]

    def random_date(self) -> Tuple[int, int, int]:
        """
        Randomly select a date from the available dates in the data.

        Returns:
            Tuple[int, int, int]: The year, month, and day of the selected date.
        """
        dates = self.list_dates()
        year, month, day = random.choice(dates)
        return year, month, day

    def split_data_set(self):
        """
        Split the data into training and testing sets based on the date.

        The first three weeks of each month are used for training and the last week for testing.
        """
        all_dates = self.list_dates()
        if not all_dates:
            self.train_dates = []
            self.test_dates = []
            return

        all_dates.sort(key=lambda x: (x[0], x[1], x[2]))  # Ensure chronological order

        train_dates: List[Tuple[int, int, int]] = []
        test_dates: List[Tuple[int, int, int]] = []

        # Group by year and month to avoid repeated logic in the loop.
        current_year, current_month = all_dates[0][0], all_dates[0][1]
        monthly_dates: List[Tuple[int, int, int]] = []

        for date in all_dates:
            year, month, _ = date
            if month != current_month or year != current_year:
                # Sort monthly dates and split into train and test
                monthly_dates.sort()
                train_len = int(len(monthly_dates) * (3 / 4))  # First three weeks for training
                train_dates += monthly_dates[:train_len]
                test_dates += monthly_dates[train_len:]

                # Reset for the new month
                monthly_dates = []
                current_month = month
                current_year = year

            monthly_dates.append(date)

        # Handle the last month
        if monthly_dates:
            monthly_dates.sort()
            train_len = int(len(monthly_dates) * (3 / 4))
            train_dates += monthly_dates[:train_len]
            test_dates += monthly_dates[train_len:]

        self.train_dates = train_dates
        self.test_dates = test_dates
