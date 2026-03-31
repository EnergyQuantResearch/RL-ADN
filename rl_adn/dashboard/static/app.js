(function () {
  const pollMs = 250;
  const statusPill = document.querySelector("[data-status-pill]");
  const episodeValue = document.querySelector("[data-episode]");
  const stepValue = document.querySelector("[data-step]");
  const feederValue = document.querySelector("[data-feeder]");
  const scenarioValue = document.querySelector("[data-scenario]");
  const priceValue = document.querySelector("[data-price]");
  const rewardValue = document.querySelector("[data-reward]");
  const dispatchSummary = document.querySelector("[data-dispatch-summary]");
  const rewardBreakdown = document.querySelector("[data-reward-breakdown]");
  const topologyTitle = document.querySelector("[data-topology-title]");
  const nodeCountValue = document.querySelector("[data-node-count]");
  const edgeCountValue = document.querySelector("[data-edge-count]");
  const canvas = document.querySelector("[data-topology-canvas]");
  const vminValue = document.querySelector("[data-vmin]");
  const vmaxValue = document.querySelector("[data-vmax]");
  const socAvgValue = document.querySelector("[data-soc-avg]");
  const dispatchTotalValue = document.querySelector("[data-dispatch-total]");
  const batterySelect = document.querySelector("[data-battery-select]");

  let selectedBattery = null;

  function edgeKey(edge) {
    return edge[0] < edge[1] ? `${edge[0]}-${edge[1]}` : `${edge[1]}-${edge[0]}`;
  }

  function createSvg(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function voltageColor(value) {
    if (value == null) {
      return "#ecf3ff";
    }
    if (value < 0.95) {
      return "#ff7b5f";
    }
    if (value > 1.05) {
      return "#ffd166";
    }
    const clamped = Math.max(0.94, Math.min(1.06, value));
    const ratio = (clamped - 0.94) / (1.06 - 0.94);
    const red = Math.round(255 - ratio * 150);
    const green = Math.round(120 + ratio * 90);
    const blue = Math.round(160 + ratio * 60);
    return `rgb(${red}, ${green}, ${blue})`;
  }

  function renderTopology(payload) {
    const latest = payload.latest;
    const layout = payload.layout;
    if (!latest || !layout) {
      canvas.innerHTML = "";
      topologyTitle.textContent = "Waiting for rollout…";
      nodeCountValue.textContent = "0 nodes";
      edgeCountValue.textContent = "0 edges";
      return;
    }

    topologyTitle.textContent = `${latest.feeder_id} / ${latest.topology_scenario}`;
    nodeCountValue.textContent = `${layout.node_count} nodes`;
    edgeCountValue.textContent = `${latest.active_edges.length} active edges`;
    canvas.innerHTML = "";

    const baselineEdgeKeys = new Set((layout.base_edges || []).map(edgeKey));
    const activeEdgeKeys = new Set((latest.active_edges || []).map(edgeKey));

    (layout.base_edges || []).forEach((edge) => {
      if (activeEdgeKeys.has(edgeKey(edge))) {
        return;
      }
      const from = layout.positions[String(edge[0])];
      const to = layout.positions[String(edge[1])];
      canvas.appendChild(createSvg("line", { class: "graph-edge is-retired", x1: from.x * 1200, y1: from.y * 760, x2: to.x * 1200, y2: to.y * 760 }));
    });

    (latest.active_edges || []).forEach((edge) => {
      const from = layout.positions[String(edge[0])];
      const to = layout.positions[String(edge[1])];
      canvas.appendChild(createSvg("line", { class: `graph-edge${baselineEdgeKeys.has(edgeKey(edge)) ? "" : " is-new"}`, x1: from.x * 1200, y1: from.y * 760, x2: to.x * 1200, y2: to.y * 760 }));
    });

    const batteryNodes = new Set((latest.battery_nodes || []).map(String));
    Object.entries(layout.positions).forEach(([nodeId, position]) => {
      const voltage = latest.node_voltages_pu[Number(nodeId) - 1];
      const circle = createSvg("circle", {
        cx: position.x * 1200,
        cy: position.y * 760,
        r: batteryNodes.has(nodeId) ? 9 : layout.node_count <= 34 ? 7 : 5.5,
        fill: voltageColor(voltage),
        stroke: batteryNodes.has(nodeId) ? "#ff7b5f" : "rgba(18, 179, 168, 0.42)",
        "stroke-width": batteryNodes.has(nodeId) ? 3 : 2.2,
      });
      circle.appendChild(createSvg("title", {}, `Node ${nodeId} | ${voltage.toFixed(4)} pu`));
      canvas.appendChild(circle);

      if (layout.node_count <= 34 || batteryNodes.has(nodeId) || nodeId === "1") {
        canvas.appendChild(createSvg("text", { class: "graph-label", x: position.x * 1200 + 10, y: position.y * 760 - 10 }, nodeId));
      }
    });
  }

  function ensureBatterySelection(latest, history) {
    const batteryNodes = latest ? latest.battery_nodes || [] : [];
    if (batteryNodes.length === 0) {
      batterySelect.innerHTML = "";
      selectedBattery = null;
      return;
    }
    if (!selectedBattery || !batteryNodes.includes(Number(selectedBattery))) {
      selectedBattery = String(batteryNodes[0]);
    }
    batterySelect.innerHTML = "";
    batteryNodes.forEach((nodeId) => {
      const option = document.createElement("option");
      option.value = String(nodeId);
      option.textContent = `Battery node ${nodeId}`;
      option.selected = option.value === selectedBattery;
      batterySelect.appendChild(option);
    });
  }

  batterySelect.addEventListener("change", () => {
    selectedBattery = batterySelect.value;
  });

  function renderMetrics(payload) {
    const latest = payload.latest;
    if (!latest) {
      statusPill.textContent = "idle";
      episodeValue.textContent = "—";
      stepValue.textContent = "—";
      feederValue.textContent = "—";
      scenarioValue.textContent = "—";
      priceValue.textContent = "—";
      rewardValue.textContent = "—";
      dispatchSummary.textContent = "No dispatch yet.";
      rewardBreakdown.textContent = "No reward available yet.";
      vminValue.textContent = "—";
      vmaxValue.textContent = "—";
      socAvgValue.textContent = "—";
      dispatchTotalValue.textContent = "—";
      return;
    }

    statusPill.textContent = payload.status;
    episodeValue.textContent = String(latest.episode_id);
    stepValue.textContent = String(latest.step_index);
    feederValue.textContent = latest.feeder_id;
    scenarioValue.textContent = latest.topology_scenario;
    priceValue.textContent = latest.price.toFixed(4);
    rewardValue.textContent = latest.reward == null ? "—" : latest.reward.toFixed(4);
    dispatchSummary.textContent = latest.battery_dispatch_kw && latest.battery_dispatch_kw.length
      ? latest.battery_dispatch_kw.map((value, index) => `B${latest.battery_nodes[index]}: ${value.toFixed(2)} kW`).join(" | ")
      : "No dispatch yet.";
    rewardBreakdown.textContent = latest.reward_breakdown
      ? `economic ${latest.reward_breakdown.economic.toFixed(4)} | voltage ${latest.reward_breakdown.voltage_penalty.toFixed(4)} | saved ${latest.reward_breakdown.saved_money.toFixed(4)}`
      : "No reward available yet.";

    const voltageMin = Math.min(...latest.node_voltages_pu);
    const voltageMax = Math.max(...latest.node_voltages_pu);
    const socAvg = latest.battery_soc.length ? latest.battery_soc.reduce((sum, value) => sum + value, 0) / latest.battery_soc.length : null;
    const dispatchTotal = (latest.battery_dispatch_kw || []).reduce((sum, value) => sum + value, 0);

    vminValue.textContent = voltageMin.toFixed(4);
    vmaxValue.textContent = voltageMax.toFixed(4);
    socAvgValue.textContent = socAvg == null ? "—" : socAvg.toFixed(4);
    dispatchTotalValue.textContent = `${dispatchTotal.toFixed(2)} kW`;
  }

  function renderCharts(payload) {
    const history = payload.history;
    const latest = payload.latest;
    ensureBatterySelection(latest, history);

    Plotly.react("reward-chart", [{ x: history.steps, y: history.reward, mode: "lines+markers", line: { color: "#12b3a8", width: 3 }, marker: { size: 6 } }], { margin: { l: 40, r: 12, t: 8, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
    Plotly.react("voltage-chart", [{ x: history.steps, y: history.voltage_min, mode: "lines", name: "min", line: { color: "#ff7b5f", width: 3 } }, { x: history.steps, y: history.voltage_max, mode: "lines", name: "max", line: { color: "#0f7994", width: 3 } }], { margin: { l: 40, r: 12, t: 8, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
    Plotly.react("dispatch-chart", [{ x: history.steps, y: history.total_dispatch_kw, mode: "lines+markers", line: { color: "#122130", width: 3 }, marker: { size: 5 } }], { margin: { l: 40, r: 12, t: 8, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });

    const socTrace = selectedBattery && history.soc_by_battery[selectedBattery] ? history.soc_by_battery[selectedBattery] : [];
    Plotly.react("soc-chart", [{ x: history.steps, y: socTrace, mode: "lines+markers", line: { color: "#ffb703", width: 3 }, marker: { size: 5 } }], { margin: { l: 40, r: 12, t: 8, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
  }

  async function refresh() {
    const response = await fetch("/api/state", { cache: "no-store" });
    const payload = await response.json();
    renderMetrics(payload);
    renderTopology(payload);
    renderCharts(payload);
  }

  refresh();
  window.setInterval(refresh, pollMs);
})();
