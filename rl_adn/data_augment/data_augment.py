from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from rl_adn.data import GeneralPowerDataManager

AugmentationMethod = Literal["GMC", "GMM", "TC"]


def _require_gmm():
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("GMM-based data augmentation requires the optional dependency 'scikit-learn'.") from exc
    return GaussianMixture


def _require_gmc():
    try:
        from copulas.multivariate import GaussianMultivariate
    except ImportError as exc:
        raise ImportError("GMC-based data augmentation requires the optional dependency 'copulas'.") from exc
    return GaussianMultivariate


def _require_tc():
    try:
        from multicopula import EllipticalCopula
    except ImportError as exc:
        raise ImportError("TC-based data augmentation requires the optional dependency 'multicopula'.") from exc
    return EllipticalCopula


class ActivePowerDataManager(GeneralPowerDataManager):
    """Specialized data manager for active-power augmentation workflows."""

    def get_active_power_data(self) -> np.ndarray:
        if not self.active_power_cols:
            raise ValueError("No active power columns were found in the dataset")

        active_power_df = self.df[self.active_power_cols].copy()
        active_power_df["day"] = self.df.index.normalize()
        active_power_df["time"] = self.df.index.time

        expected_time_steps = int(24 * 60 / self.time_interval)
        complete_days = active_power_df.groupby("day").size()
        complete_days = complete_days[complete_days == expected_time_steps].index
        active_power_df = active_power_df[active_power_df["day"].isin(complete_days)]
        if active_power_df.empty:
            raise ValueError("No complete days were found for active-power augmentation")

        reshaped = active_power_df.set_index(["day", "time"]).stack().unstack("time")
        active_power_array = reshaped.to_numpy(dtype=float)
        active_power_array = active_power_array[~np.isnan(active_power_array).any(axis=1)]
        if active_power_array.size == 0:
            raise ValueError("Active power data contains only NaN values after reshaping")
        return active_power_array


class TimeSeriesDataAugmentor:
    """Generate synthetic active-power time series using copula or GMM methods."""

    def __init__(self, data: ActivePowerDataManager, augmentation_model_name: AugmentationMethod = "GMC"):
        self.data = data
        self.augmentation_model_name = augmentation_model_name
        self.augmentation_model = None
        self.n_models = int(24.0 * 60.0 / self.data.time_interval)
        self.gmm_models = []
        self._create_augmentation_model()

    def _create_augmentation_model(self) -> None:
        active_power_array = self.data.get_active_power_data()
        GaussianMixture = _require_gmm()
        best_components = [self._bic_value(active_power_array[:, i].reshape(-1, 1), 20, GaussianMixture) for i in range(self.n_models)]
        self.gmm_models = [GaussianMixture(n_components=count).fit(active_power_array[:, i].reshape(-1, 1)) for i, count in enumerate(best_components)]

        if self.augmentation_model_name == "GMC":
            GaussianMultivariate = _require_gmc()
            standard_input_data = np.empty((active_power_array.shape[0], self.n_models))
            for index in range(self.n_models):
                standard_input_data[:, index] = np.array([self._gmm_cdf(self.gmm_models[index], value) for value in active_power_array[:, index]])
            copula = GaussianMultivariate()
            copula.fit(standard_input_data)
            self.augmentation_model = copula
        elif self.augmentation_model_name == "GMM":
            self.augmentation_model = self.gmm_models
        elif self.augmentation_model_name == "TC":
            EllipticalCopula = _require_tc()
            tc_model = EllipticalCopula(active_power_array.T)
            tc_model.fit()
            self.augmentation_model = tc_model
        else:
            raise ValueError(f"Unsupported augmentation_model_name: {self.augmentation_model_name}")

    def _gmm_cdf(self, gmm, value: float) -> float:
        cdf = 0.0
        for component in range(gmm.n_components):
            cdf += gmm.weights_[component] * norm.cdf(
                value,
                gmm.means_[component, 0],
                np.sqrt(gmm.covariances_[component, 0]),
            )
        return float(cdf)

    def _inverse_gmm_cdf(self, gmm, percentile: float) -> float:
        def root_fn(x: float) -> float:
            return self._gmm_cdf(gmm, x) - percentile

        return float(brentq(root_fn, -3000, 3000))

    @staticmethod
    def _bic_value(data: np.ndarray, n_components_range: int, gaussian_mixture_cls) -> int:
        bic_values = []
        for n_components in range(1, n_components_range):
            gmm = gaussian_mixture_cls(n_components=n_components).fit(data)
            bic_values.append(gmm.bic(data))
        return int(np.argmin(bic_values) + 1)

    def augment_data(self, num_nodes: int, num_days: int, start_date: datetime) -> pd.DataFrame:
        num_samples = num_days * num_nodes
        if self.augmentation_model_name == "GMC":
            generated_pseudo_obs = self._sample_gmc(num_samples)
            transformed_samples = np.empty_like(generated_pseudo_obs)
            for index in range(self.n_models):
                transformed_samples[:, index] = np.array([self._inverse_gmm_cdf(self.gmm_models[index], u) for u in generated_pseudo_obs[:, index]])
            flattened_samples = transformed_samples.flatten()
        elif self.augmentation_model_name == "GMM":
            gmm_samples = np.empty((num_samples, self.n_models))
            for index in range(self.n_models):
                gmm_samples[:, index] = self.gmm_models[index].sample(num_samples)[0].reshape(-1)
            flattened_samples = gmm_samples.flatten()
        elif self.augmentation_model_name == "TC":
            tc_samples = self._sample_tc(num_samples)
            flattened_samples = tc_samples.flatten()
        else:
            raise ValueError(f"Unsupported augmentation_model_name: {self.augmentation_model_name}")

        timestamps = []
        node_index = []
        time_step = timedelta(minutes=self.data.time_interval)
        for day_offset in range(num_days):
            for node_id in range(1, num_nodes + 1):
                timestamps.extend([start_date + timedelta(days=day_offset) + slot * time_step for slot in range(self.n_models)])
                node_index.extend([f"active_power_node_{node_id}" for _ in range(self.n_models)])

        synthetic_data_df = pd.DataFrame({"date_time": timestamps, "node": node_index, "value": flattened_samples})
        augmented_df = synthetic_data_df.pivot(index="date_time", columns="node", values="value").reset_index()
        ordered_columns = ["date_time"] + self.sort_columns(augmented_df.columns, r"active_power(_\w+)?")
        return augmented_df[ordered_columns]

    def _sample_gmc(self, sample_count: int) -> np.ndarray:
        generated_pseudo_obs = np.empty((0, self.n_models))
        while generated_pseudo_obs.shape[0] < sample_count:
            sampled = np.array(self.augmentation_model.sample(1))
            if sampled.min() > 0 and sampled.max() < 1:
                generated_pseudo_obs = np.vstack((sampled, generated_pseudo_obs))
        return generated_pseudo_obs[:sample_count]

    def _sample_tc(self, sample_count: int) -> np.ndarray:
        tc_samples = np.empty((0, self.n_models))
        while tc_samples.shape[0] < sample_count:
            sampled = np.array(self.augmentation_model.sample(1)).reshape(1, -1)
            if not np.isinf(sampled).any():
                tc_samples = np.vstack((sampled, tc_samples))
        return tc_samples[:sample_count]

    @staticmethod
    def save_augmented_data(augmented_df: pd.DataFrame, file_name: str | Path) -> Path:
        output_path = Path(file_name)
        augmented_df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def sort_columns(columns, pattern: str):
        def sort_key(column_name: str) -> int:
            parts = column_name.split("_")
            if parts[-1].isdigit():
                return int(parts[-1])
            return 0

        filtered_cols = [column for column in columns if re.fullmatch(pattern, column)]
        return sorted(filtered_cols, key=sort_key)
