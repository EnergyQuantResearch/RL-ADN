from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _require_pyomo():
    try:
        from pyomo.environ import ConcreteModel, Constraint, Objective, Param, Set, Var, minimize
    except ImportError as exc:
        raise ImportError("Pyomo benchmark support requires the optional dependency 'pyomo'.") from exc

    return ConcreteModel, Constraint, Objective, Param, Set, Var, minimize


@dataclass(frozen=True)
class BatterySpec:
    capacity_mwh: float = 1.0
    max_charge_mw: float = 0.3
    max_discharge_mw: float = 0.3
    efficiency: float = 1.0
    degradation_eur_per_kw: float = 0.0
    max_soc: float = 0.8
    min_soc: float = 0.2
    initial_soc: float = 0.4
    time_interval_minutes: float = 15.0


@dataclass(frozen=True)
class DispatchBenchmarkData:
    times: tuple[int, ...]
    nodes: tuple[int, ...]
    lines: tuple[tuple[int, int], ...]
    tb: dict[int, int]
    pd: np.ndarray
    qd: np.ndarray
    r: dict[tuple[int, int], float]
    x: dict[tuple[int, int], float]
    battery_nodes: frozenset[int]
    price: np.ndarray
    battery: BatterySpec = field(default_factory=BatterySpec)

    @classmethod
    def from_mapping(
        cls,
        data_network: Mapping[str, Any],
        *,
        battery: BatterySpec | None = None,
    ) -> "DispatchBenchmarkData":
        times = tuple(data_network["TIMES"])
        nodes = tuple(data_network["NODES"])
        lines = tuple(data_network["LINES"])
        tb = dict(data_network["Tb"])
        pd_array = np.asarray(data_network["PD"], dtype=float)
        qd_input = data_network.get("QD")
        qd_array = np.zeros_like(pd_array) if qd_input is None else np.asarray(qd_input, dtype=float)
        price = np.asarray(data_network["PRICE"], dtype=float)
        if pd_array.shape[0] != len(times):
            raise ValueError("PD must have one row per time step")
        if qd_array.shape != pd_array.shape:
            raise ValueError("QD must match PD shape")
        if price.shape[0] != len(times):
            raise ValueError("PRICE must have one value per time step")
        return cls(
            times=times,
            nodes=nodes,
            lines=lines,
            tb=tb,
            pd=pd_array,
            qd=qd_array,
            r=dict(data_network["R"]),
            x=dict(data_network["X"]),
            battery_nodes=frozenset(data_network["BATTERY_NODES"]),
            price=price,
            battery=battery or BatterySpec(),
        )


def construct_opf_model(v_nom: float, v_min: float, v_max: float, data_network: Mapping[str, Any] | DispatchBenchmarkData):
    """Build the Pyomo dispatch model for ESS scheduling on a radial feeder."""
    ConcreteModel, Constraint, Objective, Param, Set, Var, minimize = _require_pyomo()
    data = data_network if isinstance(data_network, DispatchBenchmarkData) else DispatchBenchmarkData.from_mapping(data_network)

    model = ConcreteModel()
    model.NODES = Set(initialize=data.nodes)
    model.LINES = Set(initialize=data.lines)
    model.TIMES = Set(initialize=data.times)

    model.Vnom = Param(initialize=v_nom, mutable=False)
    model.Vmin = Param(initialize=v_min, mutable=False)
    model.Vmax = Param(initialize=v_max, mutable=False)
    model.Tb = Param(model.NODES, initialize=data.tb, mutable=True)
    model.QD = Param(
        model.TIMES,
        model.NODES,
        initialize=lambda _, time, node: float(data.qd[time, node]),
        mutable=False,
    )
    model.R = Param(model.LINES, initialize=data.r, mutable=False)
    model.X = Param(model.LINES, initialize=data.x, mutable=False)
    model.battery_initial_soc = Param(default=data.battery.initial_soc)
    model.battery_capacity = Param(default=data.battery.capacity_mwh)
    model.battery_soc_max = Param(default=data.battery.max_soc)
    model.battery_soc_min = Param(default=data.battery.min_soc)
    model.battery_max_change = Param(default=data.battery.max_charge_mw)
    model.PD = Param(
        model.TIMES,
        model.NODES,
        initialize=lambda _, time, node: float(data.pd[time, node]),
    )
    model.RM = Param(model.LINES, initialize=lambda _, i, j: model.R[i, j])
    model.XM = Param(model.LINES, initialize=lambda _, i, j: model.X[i, j])

    model.P = Var(model.TIMES, model.LINES, initialize=0)
    model.Q = Var(model.TIMES, model.LINES, initialize=0)
    model.I = Var(model.TIMES, model.LINES, initialize=0)
    model.SOC = Var(
        model.TIMES,
        model.NODES,
        initialize=model.battery_initial_soc,
        bounds=(model.battery_soc_min, model.battery_soc_max),
    )

    def energy_change_rule(model, time, node):
        model.energy_change[time, node].fixed = node not in data.battery_nodes
        return 0.0

    model.energy_change = Var(
        model.TIMES,
        model.NODES,
        initialize=energy_change_rule,
        bounds=(-model.battery_max_change, model.battery_max_change),
    )

    def substation_active_rule(model, time, node):
        if model.Tb[node].value == 0:
            model.PS[time, node].fixed = True
        return 0.0

    def substation_reactive_rule(model, time, node):
        if model.Tb[node].value == 0:
            model.QS[time, node].fixed = True
        return 0.0

    model.PS = Var(model.TIMES, model.NODES, initialize=substation_active_rule)
    model.QS = Var(model.TIMES, model.NODES, initialize=substation_reactive_rule)
    model.PRICE = Param(model.TIMES, initialize=lambda _, time: float(data.price[time]), mutable=False)

    def voltage_init(model, time, node):
        if model.Tb[node].value == 1:
            model.V[time, node].fixed = True
        return model.Vnom

    model.V = Var(model.TIMES, model.NODES, initialize=voltage_init)
    model.obj = Objective(
        rule=lambda model: sum(
            sum(model.PS[time, node] * model.PRICE[time] for node in model.NODES)
            for time in model.TIMES
        ),
        sense=minimize,
    )

    interval_hours = data.battery.time_interval_minutes / 60.0

    def soc_update_rule(model, time, node):
        if node not in data.battery_nodes:
            return Constraint.Skip
        if time == model.TIMES.first():
            return model.SOC[time, node] == model.battery_initial_soc - (
                model.energy_change[time, node] * interval_hours
            ) / model.battery_capacity
        return model.SOC[time, node] == model.SOC[model.TIMES.prev(time), node] - (
            model.energy_change[time, node] * interval_hours
        ) / model.battery_capacity

    model.constaint_soc_update = Constraint(model.TIMES, model.NODES, rule=soc_update_rule)

    def active_power_flow_rule(model, time, node):
        return (
            sum(model.P[time, (j, i)] for j, i in model.LINES if i == node)
            - sum(
                model.P[time, (i, j)] + model.RM[i, j] * (model.I[time, (i, j)] ** 2)
                for i, j in model.LINES
                if node == i
            )
            + model.PS[time, node]
            + model.energy_change[time, node]
            == model.PD[time, node]
        )

    def reactive_power_flow_rule(model, time, node):
        return (
            sum(model.Q[time, (j, i)] for j, i in model.LINES if i == node)
            - sum(
                model.Q[time, (i, j)] + model.XM[i, j] * (model.I[time, (i, j)] ** 2)
                for i, j in model.LINES
                if node == i
            )
            + model.QS[time, node]
            == model.QD[time, node]
        )

    def voltage_drop_rule(model, time, i, j):
        return (
            model.V[time, i] ** 2
            - 2 * (model.RM[i, j] * model.P[time, (i, j)] + model.XM[i, j] * model.Q[time, (i, j)])
            - (model.RM[i, j] ** 2 + model.XM[i, j] ** 2) * model.I[time, (i, j)] ** 2
            - model.V[time, j] ** 2
        ) == 0

    def current_definition_rule(model, time, i, j):
        return (model.I[time, (i, j)] ** 2) * (model.V[time, j] ** 2) == (
            model.P[time, (i, j)] ** 2 + model.Q[time, (i, j)] ** 2
        )

    model.active_power_flow = Constraint(model.TIMES, model.NODES, rule=active_power_flow_rule)
    model.reactive_power_flow = Constraint(model.TIMES, model.NODES, rule=reactive_power_flow_rule)
    model.voltage_drop = Constraint(model.TIMES, model.LINES, rule=voltage_drop_rule)
    model.define_current = Constraint(model.TIMES, model.LINES, rule=current_definition_rule)
    model.current_limit = Constraint(model.TIMES, model.LINES, rule=lambda model, time, i, j: (0, model.I[time, (i, j)], None))
    model.voltage_limit = Constraint(
        model.TIMES,
        model.NODES,
        rule=lambda model, time, node: (model.Vmin, model.V[time, node], model.Vmax)
        if node in data.battery_nodes
        else Constraint.Skip,
    )
    return model


def convert_indexed_values_to_frame(data: Mapping[tuple[int, int], Any]) -> pd.DataFrame:
    columns = sorted({key[1] for key in data})
    frame = pd.DataFrame(columns=columns)
    for (row_index, column_index), value in data.items():
        frame.loc[row_index, column_index] = value
    return frame.sort_index().sort_index(axis=1)


def convert_dict_to_pd(data: Mapping[tuple[int, int], Any]) -> pd.DataFrame:
    """Legacy alias for ``convert_indexed_values_to_frame``."""
    return convert_indexed_values_to_frame(data)
