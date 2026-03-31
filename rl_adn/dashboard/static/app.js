(function () {
  const pollMs = 250;
  const translations = {
    zh: {
      heroEyebrow: "RL-ADN / agentic grid console",
      heroTitle: "主动配电网的实时单线图监控屏。",
      heroLead: "让拓扑变化、节点电压、电池 dispatch 与 reward 在同一块控制屏里被看见，并由系统自动解释。",
      language: "语言",
      historyWindow: "历史窗口",
      localBrowser: "本地浏览器",
      singleEnv: "单环境 v1",
      scenarioStatus: "场景状态",
      selectedNode: "当前节点",
      selectedBattery: "当前电池",
      runState: "运行状态",
      episode: "轮次",
      step: "步数",
      feeder: "馈线",
      scenario: "场景",
      price: "价格",
      reward: "奖励",
      control: "控制动作",
      rewardBreakdown: "奖励拆解",
      interaction: "交互控制",
      batteryFocus: "电池焦点",
      chartMode: "信号视图",
      topology: "拓扑图",
      telemetry: "遥测指标",
      voltageMin: "最低电压",
      voltageMax: "最高电压",
      socAvg: "平均 SOC",
      totalDispatch: "总 dispatch",
      selectedDispatch: "焦点 dispatch",
      selectedSoc: "焦点 SOC",
      priceRewardFocus: "价格 / 奖励聚焦",
      rewardPriceChart: "奖励与价格时序",
      voltageChart: "电压区间",
      dispatchChart: "选中电池 dispatch",
      socChart: "选中电池 SOC",
      legendNominal: "正常支路",
      legendRewired: "重构支路",
      legendRetired: "退役支路",
      legendBattery: "电池节点",
      agentNarration: "Agent 自动旁白",
      agentNarrationTitle: "系统解释",
      narrationAction: "动作",
      narrationReason: "原因",
      narrationRisk: "风险",
      waiting: "等待 rollout...",
      noScenario: "暂无激活拓扑。",
      noNode: "暂无节点遥测。",
      noBattery: "暂无选中电池。",
      noDispatch: "暂无 dispatch。",
      noReward: "暂无奖励信息。",
      noPriceTrend: "暂无价格与奖励趋势。",
      nodeLabel: "节点",
      batteryLabel: "电池节点",
      running: "运行中",
      finished: "已结束",
      idle: "空闲",
      nodes: "节点",
      activeEdges: "激活边",
      split: "分离",
      overlay: "叠加",
      active: "在役",
      retired: "退役",
    },
    en: {
      heroEyebrow: "RL-ADN / agentic grid console",
      heroTitle: "Real-time single-line control console for active distribution networks.",
      heroLead: "Bring topology shifts, node voltages, battery dispatch, and reward onto one operator-facing screen with automatic explanations.",
      language: "Language",
      historyWindow: "History",
      localBrowser: "local browser",
      singleEnv: "single-env v1",
      scenarioStatus: "Scenario status",
      selectedNode: "Selected node",
      selectedBattery: "Selected battery",
      runState: "Run state",
      episode: "Episode",
      step: "Step",
      feeder: "Feeder",
      scenario: "Scenario",
      price: "Price",
      reward: "Reward",
      control: "Control",
      rewardBreakdown: "Reward breakdown",
      interaction: "Interaction",
      batteryFocus: "Battery focus",
      chartMode: "Signal view",
      topology: "Topology",
      telemetry: "Telemetry",
      voltageMin: "Voltage min",
      voltageMax: "Voltage max",
      socAvg: "SOC avg",
      totalDispatch: "Total dispatch",
      selectedDispatch: "Selected dispatch",
      selectedSoc: "Selected SOC",
      priceRewardFocus: "Price / reward focus",
      rewardPriceChart: "Reward & price over time",
      voltageChart: "Voltage band",
      dispatchChart: "Selected battery dispatch",
      socChart: "Selected battery SOC",
      legendNominal: "nominal edge",
      legendRewired: "rewired edge",
      legendRetired: "retired edge",
      legendBattery: "battery node",
      agentNarration: "Agent narration",
      agentNarrationTitle: "System explanation",
      narrationAction: "Action",
      narrationReason: "Reason",
      narrationRisk: "Risk",
      waiting: "Waiting for rollout...",
      noScenario: "No active topology yet.",
      noNode: "No node telemetry yet.",
      noBattery: "No battery selected.",
      noDispatch: "No dispatch yet.",
      noReward: "No reward available yet.",
      noPriceTrend: "No price-reward trend yet.",
      nodeLabel: "Node",
      batteryLabel: "Battery node",
      running: "running",
      finished: "finished",
      idle: "idle",
      nodes: "nodes",
      activeEdges: "active edges",
      split: "Split",
      overlay: "Overlay",
      active: "active",
      retired: "retired",
    },
  };

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
  const topologyPanelTitle = document.querySelector("[data-topology-panel-title]");
  const nodeCountValue = document.querySelector("[data-node-count]");
  const edgeCountValue = document.querySelector("[data-edge-count]");
  const scenarioDescription = document.querySelector("[data-scenario-description]");
  const canvas = document.querySelector("[data-topology-canvas]");
  const vminValue = document.querySelector("[data-vmin]");
  const vmaxValue = document.querySelector("[data-vmax]");
  const socAvgValue = document.querySelector("[data-soc-avg]");
  const dispatchTotalValue = document.querySelector("[data-dispatch-total]");
  const selectedDispatchValue = document.querySelector("[data-selected-dispatch]");
  const selectedSocValue = document.querySelector("[data-selected-soc]");
  const batterySelect = document.querySelector("[data-battery-select]");
  const languageSelect = document.querySelector("[data-language-select]");
  const windowSelect = document.querySelector("[data-window-select]");
  const chartModeSelect = document.querySelector("[data-chart-mode-select]");
  const selectedNodeLabel = document.querySelector("[data-selected-node]");
  const selectedNodeDetail = document.querySelector("[data-selected-node-detail]");
  const selectedBatteryLabel = document.querySelector("[data-selected-battery]");
  const selectedBatteryDetail = document.querySelector("[data-selected-battery-detail]");
  const priceRewardSummary = document.querySelector("[data-price-reward-summary]");
  const narrationAction = document.querySelector("[data-narration-action]");
  const narrationReason = document.querySelector("[data-narration-reason]");
  const narrationRisk = document.querySelector("[data-narration-risk]");

  let locale = "zh";
  let historyWindow = "all";
  let chartMode = "split";
  let selectedBattery = null;
  let selectedNode = 1;
  let lastPayload = null;

  function t(key) {
    return translations[locale][key] || translations.zh[key] || key;
  }

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

  function appendBatteryIcon(group, x, y) {
    const batteryGroup = createSvg("g", {
      transform: `translate(${x + 12}, ${y - 28})`,
      class: "graph-battery-icon",
    });
    batteryGroup.appendChild(createSvg("rect", { class: "graph-battery-shell", x: 0, y: 0, width: 18, height: 28, rx: 4, ry: 4 }));
    batteryGroup.appendChild(createSvg("rect", { class: "graph-battery-cap", x: 5.5, y: -4, width: 7, height: 4, rx: 1.5, ry: 1.5 }));
    batteryGroup.appendChild(createSvg("path", { class: "graph-battery-bolt", d: "M8 4 L4.5 14 H8.4 L6.6 24 L13.5 11.5 H9.6 L12.2 4 Z" }));
    group.appendChild(batteryGroup);
  }

  function createBatteryCard(group, x, y, meta, soc, dispatch) {
    const anchor = meta.battery_card_anchor || "start";
    const cardX = x + meta.battery_card_dx;
    const cardY = y + meta.battery_card_dy;
    const width = 110;
    const height = 36;
    const rectX = anchor === "end" ? cardX - width : cardX;
    const textX = anchor === "end" ? cardX - 10 : cardX + 10;
    const card = createSvg("g", { class: "graph-battery-card-group" });
    card.appendChild(createSvg("rect", { class: "graph-battery-card", x: rectX, y: cardY, width, height, rx: 10, ry: 10 }));
    card.appendChild(createSvg("text", { class: "graph-battery-card-text", x: textX, y: cardY + 15, "text-anchor": anchor }, `SOC ${soc.toFixed(2)}`));
    card.appendChild(createSvg("text", { class: "graph-battery-card-text", x: textX, y: cardY + 29, "text-anchor": anchor }, `${dispatch.toFixed(1)} kW`));
    group.appendChild(card);
  }

  function getNodeMeta(layout, nodeId) {
    return (layout.node_meta && layout.node_meta[String(nodeId)]) || {
      anchor: "middle",
      label_dx: 0,
      label_dy: -18,
      metric_dx: 0,
      metric_dy: 24,
      battery_icon_dx: 12,
      battery_icon_dy: -28,
      battery_card_dx: 12,
      battery_card_dy: 42,
      battery_card_anchor: "start",
    };
  }

  function voltageColor(value) {
    if (value == null) return "#dbe9ff";
    if (value < 0.95) return "#ff7b5f";
    if (value > 1.05) return "#ffd166";
    const clamped = Math.max(0.94, Math.min(1.06, value));
    const ratio = (clamped - 0.94) / (1.06 - 0.94);
    const red = Math.round(94 + ratio * 28);
    const green = Math.round(167 + ratio * 48);
    const blue = Math.round(181 + ratio * 32);
    return `rgb(${red}, ${green}, ${blue})`;
  }

  function applyTranslations() {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    chartModeSelect.options[0].textContent = t("split");
    chartModeSelect.options[1].textContent = t("overlay");
    if (lastPayload) render(lastPayload);
  }

  function trimHistory(history) {
    if (historyWindow === "all") return history;
    const keep = Number(historyWindow);
    const start = Math.max(history.steps.length - keep, 0);
    return {
      steps: history.steps.slice(start),
      reward: history.reward.slice(start),
      price: history.price.slice(start),
      voltage_min: history.voltage_min.slice(start),
      voltage_max: history.voltage_max.slice(start),
      total_dispatch_kw: history.total_dispatch_kw.slice(start),
      soc_by_battery: Object.fromEntries(Object.entries(history.soc_by_battery).map(([key, value]) => [key, value.slice(start)])),
      dispatch_by_battery: Object.fromEntries(Object.entries(history.dispatch_by_battery).map(([key, value]) => [key, value.slice(start)])),
      scenario_markers: history.scenario_markers.filter((marker) => history.steps.slice(start).includes(marker.step)),
    };
  }

  function ensureBatterySelection(latest) {
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
      option.textContent = `${t("batteryLabel")} ${nodeId}`;
      option.selected = option.value === selectedBattery;
      batterySelect.appendChild(option);
    });
  }

  function renderTopology(payload) {
    const latest = payload.latest;
    const layout = payload.layout;
    if (!latest || !layout) {
      canvas.innerHTML = "";
      topologyTitle.textContent = t("waiting");
      topologyPanelTitle.textContent = t("waiting");
      nodeCountValue.textContent = `0 ${t("nodes")}`;
      edgeCountValue.textContent = `0 ${t("activeEdges")}`;
      scenarioDescription.textContent = t("noScenario");
      return;
    }

    topologyTitle.textContent = `${latest.feeder_id} / ${latest.topology_scenario}`;
    topologyPanelTitle.textContent = `${latest.feeder_id} / ${latest.topology_scenario}`;
    nodeCountValue.textContent = `${layout.node_count} ${t("nodes")}`;
    edgeCountValue.textContent = `${latest.active_edges.length} ${t("activeEdges")}`;
    scenarioDescription.textContent = payload.narration ? payload.narration[locale].risk : latest.topology_scenario;
    canvas.innerHTML = "";

    const baselineEdgeKeys = new Set((layout.base_edges || []).map(edgeKey));
    const activeEdgeKeys = new Set((latest.active_edges || []).map(edgeKey));
    const batteryNodes = new Set((latest.battery_nodes || []).map(String));

    (layout.base_edges || []).forEach((edge) => {
      if (activeEdgeKeys.has(edgeKey(edge))) return;
      const from = layout.positions[String(edge[0])];
      const to = layout.positions[String(edge[1])];
      canvas.appendChild(createSvg("line", { class: "graph-edge is-retired", x1: from.x * 1400, y1: from.y * 820, x2: to.x * 1400, y2: to.y * 820 }));
      const midX = ((from.x + to.x) * 1400) / 2;
      const midY = ((from.y + to.y) * 820) / 2;
      canvas.appendChild(createSvg("text", { class: "graph-node-metric", x: midX + 6, y: midY - 4 }, t("retired")));
    });

    (latest.active_edges || []).forEach((edge) => {
      const from = layout.positions[String(edge[0])];
      const to = layout.positions[String(edge[1])];
      const isNew = !baselineEdgeKeys.has(edgeKey(edge));
      canvas.appendChild(createSvg("line", { class: `graph-edge${isNew ? " is-new" : ""}`, x1: from.x * 1400, y1: from.y * 820, x2: to.x * 1400, y2: to.y * 820 }));
      if (isNew) {
        const midX = ((from.x + to.x) * 1400) / 2;
        const midY = ((from.y + to.y) * 820) / 2;
        canvas.appendChild(createSvg("text", { class: "graph-node-metric", x: midX + 6, y: midY - 4 }, t("active")));
      }
    });

    Object.entries(layout.positions).forEach(([nodeId, position]) => {
      const voltage = latest.node_voltages_pu[Number(nodeId) - 1];
      const isBattery = batteryNodes.has(nodeId);
      const isSelected = Number(nodeId) === selectedNode;
      const meta = getNodeMeta(layout, nodeId);
      const x = position.x * 1400;
      const y = position.y * 820;
      const group = createSvg("g", { class: "graph-node-group" });
      const circle = createSvg("circle", {
        class: `graph-node${isSelected ? " is-selected" : ""}`,
        cx: x,
        cy: y,
        r: isBattery ? 10 : layout.node_count <= 34 ? 7.5 : 6,
        fill: voltageColor(voltage),
        stroke: isBattery ? "#ff7b5f" : "rgba(22, 217, 194, 0.42)",
        "stroke-width": isBattery ? 3.2 : 2.2,
      });
      circle.appendChild(createSvg("title", {}, `${t("nodeLabel")} ${nodeId} | ${voltage.toFixed(4)} pu`));
      circle.addEventListener("click", () => {
        selectedNode = Number(nodeId);
        if (isBattery) selectedBattery = nodeId;
        render(lastPayload);
      });
      group.appendChild(circle);
      if (isBattery) appendBatteryIcon(group, x + meta.battery_icon_dx - 12, y + meta.battery_icon_dy + 28);
      canvas.appendChild(group);

      canvas.appendChild(
        createSvg(
          "text",
          {
            class: "graph-label",
            x: x + meta.label_dx,
            y: y + meta.label_dy,
            "text-anchor": meta.anchor,
          },
          nodeId
        )
      );
      canvas.appendChild(
        createSvg(
          "text",
          {
            class: "graph-node-metric",
            x: x + meta.metric_dx,
            y: y + meta.metric_dy,
            "text-anchor": meta.anchor,
          },
          `|V| ${voltage.toFixed(3)}`
        )
      );

      if (isBattery) {
        const batteryIndex = latest.battery_nodes.indexOf(Number(nodeId));
        const batteryDispatch = latest.battery_dispatch_kw ? latest.battery_dispatch_kw[batteryIndex] : null;
        const batterySoc = latest.battery_soc[batteryIndex];
        if (batteryDispatch != null) {
          createBatteryCard(canvas, x, y, meta, batterySoc, batteryDispatch);
        }
      }
    });
  }

  function renderMetrics(payload) {
    const latest = payload.latest;
    if (!latest) {
      statusPill.textContent = t("idle");
      episodeValue.textContent = "-";
      stepValue.textContent = "-";
      feederValue.textContent = "-";
      scenarioValue.textContent = "-";
      priceValue.textContent = "-";
      rewardValue.textContent = "-";
      dispatchSummary.textContent = t("noDispatch");
      rewardBreakdown.textContent = t("noReward");
      vminValue.textContent = "-";
      vmaxValue.textContent = "-";
      socAvgValue.textContent = "-";
      dispatchTotalValue.textContent = "-";
      selectedDispatchValue.textContent = "-";
      selectedSocValue.textContent = "-";
      selectedNodeLabel.textContent = `${t("nodeLabel")} ${selectedNode}`;
      selectedNodeDetail.textContent = t("noNode");
      selectedBatteryLabel.textContent = "-";
      selectedBatteryDetail.textContent = t("noBattery");
      narrationAction.textContent = t("waiting");
      narrationReason.textContent = t("noReward");
      narrationRisk.textContent = t("noScenario");
      priceRewardSummary.textContent = t("noPriceTrend");
      return;
    }

    statusPill.textContent = t(payload.status);
    episodeValue.textContent = String(latest.episode_id);
    stepValue.textContent = String(latest.step_index);
    feederValue.textContent = latest.feeder_id;
    scenarioValue.textContent = latest.topology_scenario;
    priceValue.textContent = latest.price.toFixed(4);
    rewardValue.textContent = latest.reward == null ? "-" : latest.reward.toFixed(4);
    dispatchSummary.textContent = latest.battery_dispatch_kw && latest.battery_dispatch_kw.length
      ? latest.battery_dispatch_kw.map((value, index) => `B${latest.battery_nodes[index]}: ${value.toFixed(2)} kW`).join(" | ")
      : t("noDispatch");
    rewardBreakdown.textContent = latest.reward_breakdown
      ? `economic ${latest.reward_breakdown.economic.toFixed(4)} | voltage ${latest.reward_breakdown.voltage_penalty.toFixed(4)} | saved ${latest.reward_breakdown.saved_money.toFixed(4)}`
      : t("noReward");

    const voltageMin = Math.min(...latest.node_voltages_pu);
    const voltageMax = Math.max(...latest.node_voltages_pu);
    const socAvg = latest.battery_soc.length ? latest.battery_soc.reduce((sum, value) => sum + value, 0) / latest.battery_soc.length : null;
    const dispatchTotal = (latest.battery_dispatch_kw || []).reduce((sum, value) => sum + value, 0);
    vminValue.textContent = voltageMin.toFixed(4);
    vmaxValue.textContent = voltageMax.toFixed(4);
    socAvgValue.textContent = socAvg == null ? "-" : socAvg.toFixed(4);
    dispatchTotalValue.textContent = `${dispatchTotal.toFixed(2)} kW`;

    const selectedVoltage = latest.node_voltages_pu[selectedNode - 1];
    selectedNodeLabel.textContent = `${t("nodeLabel")} ${selectedNode}`;
    selectedNodeDetail.textContent = selectedVoltage == null ? t("noNode") : `${selectedVoltage.toFixed(4)} pu`;

    const batteryIndex = selectedBattery ? latest.battery_nodes.indexOf(Number(selectedBattery)) : -1;
    if (batteryIndex >= 0) {
      const batteryDispatch = latest.battery_dispatch_kw ? latest.battery_dispatch_kw[batteryIndex] : null;
      const batterySoc = latest.battery_soc[batteryIndex];
      selectedDispatchValue.textContent = batteryDispatch == null ? "-" : `${batteryDispatch.toFixed(2)} kW`;
      selectedSocValue.textContent = batterySoc == null ? "-" : batterySoc.toFixed(4);
      selectedBatteryLabel.textContent = `${t("batteryLabel")} ${selectedBattery}`;
      selectedBatteryDetail.textContent = `SOC ${batterySoc.toFixed(4)}${batteryDispatch == null ? "" : ` | ${batteryDispatch.toFixed(2)} kW`}`;
    } else {
      selectedDispatchValue.textContent = "-";
      selectedSocValue.textContent = "-";
      selectedBatteryLabel.textContent = "-";
      selectedBatteryDetail.textContent = t("noBattery");
    }

    if (payload.narration) {
      narrationAction.textContent = payload.narration[locale].action;
      narrationReason.textContent = payload.narration[locale].reason;
      narrationRisk.textContent = payload.narration[locale].risk;
    } else {
      narrationAction.textContent = t("waiting");
      narrationReason.textContent = t("noReward");
      narrationRisk.textContent = t("noScenario");
    }

    priceRewardSummary.textContent = latest.reward == null ? t("noPriceTrend") : `price ${latest.price.toFixed(4)} | reward ${latest.reward.toFixed(4)} | ${latest.topology_scenario}`;
  }

  function renderCharts(payload) {
    const history = trimHistory(payload.history);
    const latest = payload.latest;
    ensureBatterySelection(latest);

    const rewardLayout = {
      margin: { l: 44, r: 44, t: 8, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d8e7f6" },
      xaxis: { gridcolor: "rgba(143,163,181,0.12)" },
      yaxis: { gridcolor: "rgba(143,163,181,0.12)" },
      legend: { orientation: "h" },
    };
    if (chartMode === "overlay") {
      rewardLayout.yaxis2 = { overlaying: "y", side: "right", showgrid: false, color: "#5cc7ff" };
    }

    Plotly.react(
      "reward-chart",
      [
        { x: history.steps, y: history.reward, mode: "lines+markers", name: t("reward"), line: { color: "#16d9c2", width: 3 }, marker: { size: 5 } },
        { x: history.steps, y: history.price, mode: chartMode === "overlay" ? "lines" : "lines+markers", name: t("price"), yaxis: chartMode === "overlay" ? "y2" : "y", line: { color: "#5cc7ff", width: 2, dash: "dot" }, marker: { size: 4 } },
      ],
      rewardLayout,
      { displayModeBar: false, responsive: true }
    );

    Plotly.react(
      "voltage-chart",
      [
        { x: history.steps, y: history.voltage_min, mode: "lines", name: "min", line: { color: "#ff7b5f", width: 3 } },
        { x: history.steps, y: history.voltage_max, mode: "lines", name: "max", line: { color: "#5cc7ff", width: 3 } },
      ],
      {
        margin: { l: 44, r: 16, t: 8, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#d8e7f6" },
        xaxis: { gridcolor: "rgba(143,163,181,0.12)" },
        yaxis: { gridcolor: "rgba(143,163,181,0.12)" },
        legend: { orientation: "h" },
      },
      { displayModeBar: false, responsive: true }
    );

    const selectedDispatch = selectedBattery && history.dispatch_by_battery[selectedBattery] ? history.dispatch_by_battery[selectedBattery] : [];
    Plotly.react(
      "dispatch-chart",
      [{ x: history.steps, y: selectedDispatch, mode: "lines+markers", line: { color: "#ffd166", width: 3 }, marker: { size: 5 } }],
      {
        margin: { l: 44, r: 16, t: 8, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#d8e7f6" },
        xaxis: { gridcolor: "rgba(143,163,181,0.12)" },
        yaxis: { gridcolor: "rgba(143,163,181,0.12)" },
      },
      { displayModeBar: false, responsive: true }
    );

    const socTrace = selectedBattery && history.soc_by_battery[selectedBattery] ? history.soc_by_battery[selectedBattery] : [];
    Plotly.react(
      "soc-chart",
      [{ x: history.steps, y: socTrace, mode: "lines+markers", line: { color: "#82f7b2", width: 3 }, marker: { size: 5 } }],
      {
        margin: { l: 44, r: 16, t: 8, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#d8e7f6" },
        xaxis: { gridcolor: "rgba(143,163,181,0.12)" },
        yaxis: { gridcolor: "rgba(143,163,181,0.12)" },
      },
      { displayModeBar: false, responsive: true }
    );
  }

  function render(payload) {
    lastPayload = payload;
    renderMetrics(payload);
    renderTopology(payload);
    renderCharts(payload);
  }

  async function refresh() {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      render(payload);
    } catch (_error) {
      return;
    }
  }

  batterySelect.addEventListener("change", () => {
    selectedBattery = batterySelect.value;
    if (lastPayload) render(lastPayload);
  });

  languageSelect.addEventListener("change", () => {
    locale = languageSelect.value;
    applyTranslations();
  });

  windowSelect.addEventListener("change", () => {
    historyWindow = windowSelect.value;
    if (lastPayload) render(lastPayload);
  });

  chartModeSelect.addEventListener("change", () => {
    chartMode = chartModeSelect.value;
    if (lastPayload) render(lastPayload);
  });

  applyTranslations();
  refresh();
  window.setInterval(refresh, pollMs);
})();
