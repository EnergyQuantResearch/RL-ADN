from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_adn.network.grid import GridTensor
from rl_adn.network.utils import create_pandapower_net


@dataclass(frozen=True)
class PowerFlowSnapshot:
    node_voltages_pu: np.ndarray
    import_power_kw: float


class LaurentSolverAdapter:
    def __init__(self, *, bus_info: pd.DataFrame, line_info: pd.DataFrame, s_base: float) -> None:
        self.node_count = len(bus_info)
        self.grid = GridTensor(
            node_file_path="",
            lines_file_path="",
            from_file=False,
            nodes_frame=bus_info.copy(deep=True),
            lines_frame=line_info.copy(deep=True),
            s_base=s_base,
        )
        self.grid.Q_file = np.zeros(self.node_count - 1)
        self.dense_ybus = self.grid._make_y_bus().toarray()

    def observe(self, net_load_kw: np.ndarray) -> PowerFlowSnapshot:
        active_power = np.asarray(net_load_kw[1:], dtype=float)
        solution = self.grid.run_pf(active_power=active_power)
        voltages = np.ones(self.node_count, dtype=np.float32)
        voltages[1:] = np.abs(solution["v"].T).reshape(-1).astype(np.float32)
        v_total = np.insert(solution["v"], 0, 1)
        import_power = float(np.matmul(self.dense_ybus, v_total)[0].real)
        return PowerFlowSnapshot(node_voltages_pu=voltages, import_power_kw=import_power)

    def dispatch(self, net_load_kw: np.ndarray, battery_nodes: tuple[int, ...], battery_power_kw: np.ndarray) -> PowerFlowSnapshot:
        adjusted_load = np.asarray(net_load_kw, dtype=float).copy()
        for node_index, dispatch_kw in zip(battery_nodes, battery_power_kw):
            adjusted_load[node_index] += float(dispatch_kw)
        return self.observe(adjusted_load)


class PandaPowerSolverAdapter:
    def __init__(self, *, network_info: dict[str, object], bus_info: pd.DataFrame, line_info: pd.DataFrame, s_base: float) -> None:
        self.network_info = dict(network_info)
        self.bus_info = bus_info.copy(deep=True)
        self.line_info = line_info.copy(deep=True)
        self.node_count = len(bus_info)
        self.s_base = s_base

    def _run_power_flow(self, net_load_kw: np.ndarray) -> PowerFlowSnapshot:
        pp, _ = _require_pandapower()
        net = create_pandapower_net(self.network_info, branch_info=self.line_info, bus_info=self.bus_info)
        net_load_mw = np.asarray(net_load_kw, dtype=float) / 1000.0
        for load_index, load_bus in enumerate(net.load.bus.values):
            net.load.p_mw.iloc[load_index] = net_load_mw[int(load_bus)]
            net.load.q_mvar.iloc[load_index] = 0.0
        pp.runpp(net, algorithm="nr")
        voltages = net.res_bus.vm_pu.to_numpy(dtype=np.float32)
        import_power = float(net.res_ext_grid["p_mw"].iloc[0] * 1000.0)
        return PowerFlowSnapshot(node_voltages_pu=voltages, import_power_kw=import_power)

    def observe(self, net_load_kw: np.ndarray) -> PowerFlowSnapshot:
        return self._run_power_flow(net_load_kw)

    def dispatch(self, net_load_kw: np.ndarray, battery_nodes: tuple[int, ...], battery_power_kw: np.ndarray) -> PowerFlowSnapshot:
        adjusted_load = np.asarray(net_load_kw, dtype=float).copy()
        for node_index, dispatch_kw in zip(battery_nodes, battery_power_kw):
            adjusted_load[node_index] += float(dispatch_kw)
        return self._run_power_flow(adjusted_load)


def _require_pandapower():
    try:
        import pandapower as pp
        import pandapower.topology as pandapower_topology
    except ImportError as exc:
        raise ImportError("PandaPower support requires the optional dependency 'pandapower'.") from exc
    return pp, pandapower_topology
