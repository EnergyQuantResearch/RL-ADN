from __future__ import annotations

from typing import Any


def _highest_dispatch(latest: dict[str, Any]) -> tuple[int | None, float]:
    dispatch = latest.get("battery_dispatch_kw") or []
    battery_nodes = latest.get("battery_nodes") or []
    if not dispatch or not battery_nodes:
        return None, 0.0
    index = max(range(len(dispatch)), key=lambda idx: abs(dispatch[idx]))
    return int(battery_nodes[index]), float(dispatch[index])


def _risk_level(latest: dict[str, Any]) -> tuple[str, str]:
    voltages = latest.get("node_voltages_pu") or []
    if not voltages:
        return "nominal", "nominal"
    voltage_min = min(voltages)
    voltage_max = max(voltages)
    if voltage_min < 0.95 or voltage_max > 1.05:
        return "critical", f"Voltage excursion detected: min {voltage_min:.4f} pu, max {voltage_max:.4f} pu."
    if voltage_min < 0.975 or voltage_max > 1.025:
        return "watch", f"Voltage margin is tightening: min {voltage_min:.4f} pu, max {voltage_max:.4f} pu."
    return "stable", f"Voltage band remains stable: min {voltage_min:.4f} pu, max {voltage_max:.4f} pu."


def _build_english(latest: dict[str, Any], history: dict[str, Any]) -> dict[str, str]:
    top_battery, top_dispatch = _highest_dispatch(latest)
    scenario = latest.get("topology_scenario", "TP1")
    reward = latest.get("reward")
    price = latest.get("price", 0.0)
    risk_level, risk_text = _risk_level(latest)

    if top_battery is None:
        action = f"System is observing {scenario} without an active battery dispatch signal yet."
    else:
        direction = "discharging" if top_dispatch > 0 else "charging"
        action = f"System is {direction} battery node {top_battery} at {abs(top_dispatch):.2f} kW under {scenario}."

    if reward is None:
        reason = f"No reward has been computed yet. Current price is {price:.4f}."
    elif reward >= 0:
        reason = f"Current reward is positive ({reward:.4f}), which suggests the dispatch is economically acceptable at price {price:.4f}."
    else:
        reason = f"Current reward is negative ({reward:.4f}), so the controller is trading off economics against grid constraints or future flexibility at price {price:.4f}."

    if scenario != "TP1":
        reason += f" Topology {scenario} is a reconfigured scenario, so the controller is adapting to a non-baseline feeder structure."

    if history["reward"]:
        recent_rewards = [value for value in history["reward"][-3:] if value is not None]
        if len(recent_rewards) >= 2 and recent_rewards[-1] < recent_rewards[0]:
            risk_text += " Reward trend has softened over the recent window."

    return {
        "action": action,
        "reason": reason,
        "risk": risk_text,
        "risk_level": risk_level,
    }


def _build_chinese(latest: dict[str, Any], history: dict[str, Any]) -> dict[str, str]:
    top_battery, top_dispatch = _highest_dispatch(latest)
    scenario = latest.get("topology_scenario", "TP1")
    reward = latest.get("reward")
    price = latest.get("price", 0.0)
    risk_level, risk_text_en = _risk_level(latest)
    voltage_min = min(latest.get("node_voltages_pu") or [1.0])
    voltage_max = max(latest.get("node_voltages_pu") or [1.0])

    if top_battery is None:
        action = f"当前系统处于 {scenario} 场景，还没有形成明确的电池控制动作。"
    else:
        direction = "放电" if top_dispatch > 0 else "充电"
        action = f"当前系统在 {scenario} 场景下，正对电池节点 {top_battery} 执行{direction}，功率约为 {abs(top_dispatch):.2f} kW。"

    if reward is None:
        reason = f"当前还没有 reward 结果，实时价格为 {price:.4f}。"
    elif reward >= 0:
        reason = f"当前 reward 为正（{reward:.4f}），说明在当前价格 {price:.4f} 下，这一步控制整体上是可接受的。"
    else:
        reason = f"当前 reward 为负（{reward:.4f}），说明系统正在经济性、约束安全或后续灵活性之间做权衡，当前价格为 {price:.4f}。"

    if scenario != "TP1":
        reason += f" 由于 {scenario} 属于重构拓扑，控制器同时在适应非基准馈线结构。"

    if risk_level == "critical":
        risk = f"当前风险较高，电压已经越界：最低 {voltage_min:.4f} pu，最高 {voltage_max:.4f} pu。"
    elif risk_level == "watch":
        risk = f"当前需要关注电压裕度，最低 {voltage_min:.4f} pu，最高 {voltage_max:.4f} pu，已经接近约束边界。"
    else:
        risk = f"当前电压区间总体稳定，最低 {voltage_min:.4f} pu，最高 {voltage_max:.4f} pu。"

    if history["reward"]:
        recent_rewards = [value for value in history["reward"][-3:] if value is not None]
        if len(recent_rewards) >= 2 and recent_rewards[-1] < recent_rewards[0]:
            risk += " 近期 reward 有走弱趋势，需要继续观察。"

    return {
        "action": action,
        "reason": reason,
        "risk": risk,
        "risk_level": risk_level,
    }


def build_narration(latest: dict[str, Any], history: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "en": _build_english(latest, history),
        "zh": _build_chinese(latest, history),
    }
