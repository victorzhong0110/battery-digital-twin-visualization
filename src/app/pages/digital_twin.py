"""第3页：数字孪生仿真器。

交互式 ECM 仿真：调节 C 倍率、温度、循环次数，实时观察放电曲线。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import load_ecm_params, get_battery_ids
from src.app.chart_theme import (
    TEMPLATE, CHART_COLORS, HEALTH_COLORS,
    BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
)


def layout() -> html.Div:
    battery_ids = get_battery_ids()
    return html.Div([
        html.Div([
            html.H1("数字孪生仿真器"),
            html.P("基于 1-RC Thevenin 等效电路模型的实时仿真与参数调节"),
        ], className="page-header"),

        # 控制面板
        html.Div([
            html.Div([
                html.Label("电池", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="twin-battery-select",
                    options=[{"label": b, "value": b} for b in battery_ids],
                    value=battery_ids[0] if battery_ids else None,
                    clearable=False, style={"width": "200px"},
                ),
            ]),
            html.Div([
                html.Label("C 倍率", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Slider(
                    id="twin-crate", min=0.5, max=3.0, step=0.25, value=1.0,
                    marks={0.5: "0.5C", 1: "1C", 1.5: "1.5C", 2: "2C", 3: "3C"},
                ),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Div([
                html.Label("环境温度 (°C)", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Slider(
                    id="twin-temp", min=10, max=50, step=5, value=25,
                    marks={10: "10°C", 25: "25°C", 35: "35°C", 45: "45°C", 50: "50°C"},
                ),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Div([
                html.Label("循环次数", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Slider(
                    id="twin-cycle", min=0, max=200, step=10, value=0,
                    marks={0: "0", 50: "50", 100: "100", 150: "150", 200: "200"},
                ),
            ], style={"flex": "1", "minWidth": "200px"}),
        ], className="card-dark", style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap",
                                         "alignItems": "flex-end", "marginBottom": "1.5rem"}),

        html.Div(id="twin-params", className="kpi-grid"),

        html.Div([
            html.Div("放电仿真", className="chart-title"),
            dcc.Graph(id="twin-discharge"),
        ], className="chart-container"),

        html.Div([
            html.Div("老化轨迹（容量 & 电阻 vs. 循环）", className="chart-title"),
            dcc.Graph(id="twin-aging"),
        ], className="chart-container"),
    ])


@callback(
    [Output("twin-params", "children"),
     Output("twin-discharge", "figure"),
     Output("twin-aging", "figure")],
    [Input("twin-battery-select", "value"),
     Input("twin-crate", "value"),
     Input("twin-temp", "value"),
     Input("twin-cycle", "value")],
)
def update_twin(battery_id: str, c_rate: float, temp_c: float, cycle: int):
    ecm_all = load_ecm_params()
    if battery_id not in ecm_all:
        empty = go.Figure().update_layout(template=TEMPLATE, height=400)
        return [], empty, empty

    params = ecm_all[battery_id]

    r0 = params["r0_initial"] + params["r0_slope"] * cycle
    r1 = params["r1_initial"] + params["r1_slope"] * cycle
    cap = max(params["capacity_initial"] + params["capacity_slope"] * cycle, 0.1)
    tau = params["tau_s"]
    c1 = tau / r1 if r1 > 0 else 100.0

    temp_coeff = params.get("temp_coeff", 0.02)
    delta_t = abs(temp_c - params.get("temp_ref_c", 24))
    r0_eff = r0 * (1 + temp_coeff * delta_t)
    r1_eff = r1 * (1 + temp_coeff * delta_t)

    soh = cap / params["rated_capacity_ah"]
    param_cards = [
        _param_kpi("R0 (有效)", f"{r0_eff * 1000:.2f} mΩ"),
        _param_kpi("R1 (有效)", f"{r1_eff * 1000:.2f} mΩ"),
        _param_kpi("剩余容量", f"{cap:.3f} Ah"),
        _param_kpi("SOH", f"{soh:.1%}"),
        _param_kpi("时间常数", f"{tau:.1f} s"),
    ]

    # ---- 放电仿真 ----
    ocv_coeffs = params["ocv_coeffs"]
    current = c_rate * params["rated_capacity_ah"]
    dt = 1.0
    max_time = int(3600 / c_rate * 1.2)

    t_arr, v_arr, soc_arr, temp_arr = [], [], [], []
    soc = 1.0
    v_rc = 0.0
    battery_temp = float(temp_c)

    for t in range(max_time):
        ocv = float(np.polyval(ocv_coeffs, soc))
        v_term = ocv - current * r0_eff - v_rc
        if v_term < 2.5 or soc <= 0:
            break
        t_arr.append(t)
        v_arr.append(v_term)
        soc_arr.append(soc)
        temp_arr.append(battery_temp)

        dv_rc = (current / c1 - v_rc / (r1_eff * c1)) * dt
        v_rc += dv_rc
        soc -= (current * dt) / (cap * 3600)

        q_gen = current ** 2 * (r0_eff + r1_eff)
        h_conv = 5.0
        a_surf = 0.01
        d_temp = (q_gen - h_conv * a_surf * (battery_temp - temp_c)) / 50.0 * dt
        battery_temp += d_temp

    t_min = np.array(t_arr) / 60
    discharge_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["端子电压", "荷电状态 (SOC)", "电池温度"],
        vertical_spacing=0.08,
    )
    discharge_fig.add_trace(go.Scatter(
        x=t_min, y=v_arr, mode="lines",
        line=dict(color=CHART_COLORS[0], width=2.5), name="电压",
    ), row=1, col=1)
    discharge_fig.add_trace(go.Scatter(
        x=t_min, y=[s * 100 for s in soc_arr], mode="lines",
        line=dict(color=CHART_COLORS[2], width=2.5), name="SOC",
    ), row=2, col=1)
    discharge_fig.add_trace(go.Scatter(
        x=t_min, y=temp_arr, mode="lines",
        line=dict(color=CHART_COLORS[3], width=2.5), name="温度",
    ), row=3, col=1)
    discharge_fig.update_yaxes(title_text="V", row=1, col=1)
    discharge_fig.update_yaxes(title_text="%", row=2, col=1)
    discharge_fig.update_yaxes(title_text="°C", row=3, col=1)
    discharge_fig.update_xaxes(title_text="时间 (分钟)", row=3, col=1)
    discharge_fig.update_layout(
        template=TEMPLATE, height=550, showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
    )

    # ---- 老化轨迹 ----
    aging_cycles = np.arange(0, 300)
    aging_cap = [max(params["capacity_initial"] + params["capacity_slope"] * c, 0.1) for c in aging_cycles]
    aging_r0 = [params["r0_initial"] + params["r0_slope"] * c for c in aging_cycles]

    aging_fig = make_subplots(specs=[[{"secondary_y": True}]])
    aging_fig.add_trace(go.Scatter(
        x=aging_cycles, y=aging_cap, mode="lines", name="容量 (Ah)",
        line=dict(color=CHART_COLORS[0], width=2.5),
    ), secondary_y=False)
    aging_fig.add_trace(go.Scatter(
        x=aging_cycles, y=[r * 1000 for r in aging_r0], mode="lines", name="R0 (mΩ)",
        line=dict(color=CHART_COLORS[3], width=2.5),
    ), secondary_y=True)
    aging_fig.add_vline(x=cycle, line_dash="dash", line_color=HEALTH_COLORS["warning"])
    aging_fig.add_annotation(x=cycle, y=1.02, yref="paper",
                              text=f"当前循环 {cycle}", showarrow=False,
                              font=dict(color=HEALTH_COLORS["warning"], size=11))
    aging_fig.update_layout(
        template=TEMPLATE, height=380, xaxis_title="循环次数",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=60, t=40, b=40),
    )
    aging_fig.update_yaxes(title_text="容量 (Ah)", secondary_y=False)
    aging_fig.update_yaxes(title_text="R0 (mΩ)", secondary_y=True)

    return param_cards, discharge_fig, aging_fig


def _param_kpi(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value", style={"fontSize": "1.3rem"}),
    ], className="kpi-card")
