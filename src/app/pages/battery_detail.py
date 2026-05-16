"""第2页：单电池深度分析。

放电曲线、特征演化、健康仪表盘。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import load_features, load_curves, get_battery_ids
from src.app.chart_theme import (
    TEMPLATE, CHART_COLORS, soh_color, soh_label,
    HEALTH_COLORS, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
)


def layout() -> html.Div:
    battery_ids = get_battery_ids()
    return html.Div([
        html.Div([
            html.H1("电池详情"),
            html.P("单电池健康状态与退化模式深度分析"),
        ], className="page-header"),

        html.Div([
            html.Div([
                html.Label("选择电池", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="detail-battery-select",
                    options=[{"label": bid, "value": bid} for bid in battery_ids],
                    value=battery_ids[0] if battery_ids else None,
                    clearable=False, style={"width": "220px"},
                ),
            ]),
        ], style={"display": "flex", "gap": "1.5rem", "marginBottom": "1.5rem", "alignItems": "flex-end"}),

        html.Div(id="detail-kpis", className="kpi-grid"),

        # 行1：SOH 仪表盘 + 容量曲线
        html.Div([
            html.Div([
                html.Div("健康状态", className="chart-title"),
                dcc.Graph(id="detail-soh-gauge"),
            ], className="chart-container", style={"flex": "0.4"}),
            html.Div([
                html.Div("容量 & SOH 随循环变化", className="chart-title"),
                dcc.Graph(id="detail-capacity-soh"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        # 行2：特征演化
        html.Div([
            html.Div("特征演化趋势", className="chart-title"),
            dcc.Graph(id="detail-features"),
        ], className="chart-container"),

        # 行3：放电电压曲线
        html.Div([
            html.Div([
                html.Span("放电电压曲线", className="chart-title"),
                html.Span("（拖动滑块对比不同循环）", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
            ]),
            dcc.Graph(id="detail-voltage-curves"),
            dcc.RangeSlider(
                id="detail-cycle-slider",
                min=0, max=10, step=1, value=[0, 5],
                marks={}, tooltip={"placement": "bottom"},
            ),
        ], className="chart-container"),
    ])


@callback(
    [Output("detail-kpis", "children"),
     Output("detail-soh-gauge", "figure"),
     Output("detail-capacity-soh", "figure"),
     Output("detail-features", "figure"),
     Output("detail-cycle-slider", "max"),
     Output("detail-cycle-slider", "value"),
     Output("detail-cycle-slider", "marks")],
    Input("detail-battery-select", "value"),
)
def update_detail(battery_id: str):
    feat = load_features()
    bdf = feat[feat["battery_id"] == battery_id].sort_values("cycle_index").reset_index(drop=True)

    if bdf.empty:
        empty_fig = go.Figure().update_layout(template=TEMPLATE, height=300)
        return [], empty_fig, empty_fig, empty_fig, 0, [0, 0], {}

    initial_cap = float(bdf["capacity_ah"].iloc[0])
    final_cap = float(bdf["capacity_ah"].iloc[-1])
    final_soh = float(bdf["soh"].iloc[-1])
    n_cycles = len(bdf)
    source = bdf["source"].iloc[0]

    health_css = "health-excellent" if final_soh > 0.90 else (
        "health-warning" if final_soh > 0.80 else "health-critical"
    )
    kpis = [
        _kpi("数据来源", source, "", ""),
        _kpi("循环总数", str(n_cycles), "", ""),
        _kpi("初始容量", f"{initial_cap:.3f} Ah", "", ""),
        _kpi("最终 SOH", f"{final_soh:.1%}", soh_label(final_soh), health_css),
        _kpi("容量衰减", f"{(1 - final_cap / initial_cap) * 100:.1f}%", "", "health-warning"),
    ]

    # SOH 仪表盘
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_soh * 100,
        number={"suffix": "%", "font": {"size": 36, "color": TEXT_PRIMARY}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_SECONDARY),
            bar=dict(color=soh_color(final_soh)),
            bgcolor=BG_CARD,
            steps=[
                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [50, 70], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [70, 80], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [80, 95], "color": "rgba(52, 211, 153, 0.1)"},
                {"range": [95, 100], "color": "rgba(16, 185, 129, 0.15)"},
            ],
            threshold=dict(line=dict(color=HEALTH_COLORS["warning"], width=2), thickness=0.75, value=80),
        ),
    ))
    gauge_fig.update_layout(template=TEMPLATE, height=300, margin=dict(l=30, r=30, t=30, b=10))

    # 容量 + SOH 双轴图
    cap_fig = make_subplots(specs=[[{"secondary_y": True}]])
    cap_fig.add_trace(go.Scatter(
        x=bdf["cycle_index"], y=bdf["capacity_ah"],
        mode="lines", name="容量 (Ah)",
        line=dict(color=CHART_COLORS[0], width=2.5),
    ), secondary_y=False)
    cap_fig.add_trace(go.Scatter(
        x=bdf["cycle_index"], y=bdf["soh"],
        mode="lines", name="SOH",
        line=dict(color=CHART_COLORS[3], width=2, dash="dot"),
    ), secondary_y=True)
    cap_fig.add_hline(y=0.80, line_dash="dash", line_color=HEALTH_COLORS["warning"], line_width=1, secondary_y=True)
    cap_fig.update_layout(
        template=TEMPLATE, height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=30, b=40),
    )
    cap_fig.update_yaxes(title_text="容量 (Ah)", secondary_y=False)
    cap_fig.update_yaxes(title_text="SOH", tickformat=".0%", secondary_y=True)
    cap_fig.update_xaxes(title_text="循环次数")

    # 特征演化 2x2
    feat_fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "内阻", "放电时长", "容量衰减率", "电阻增长率",
    ])
    feature_pairs = [
        ("internal_resistance_ohm", "Ω"),
        ("discharge_duration_s", "s"),
        ("capacity_fade_rate", "Ah/cycle"),
        ("resistance_increase_rate", "Ω/cycle"),
    ]
    for idx, (col, unit) in enumerate(feature_pairs):
        r, c = idx // 2 + 1, idx % 2 + 1
        if col in bdf.columns:
            vals = bdf[col].dropna()
            feat_fig.add_trace(go.Scatter(
                x=bdf.loc[vals.index, "cycle_index"], y=vals,
                mode="lines", line=dict(color=CHART_COLORS[idx], width=2),
                showlegend=False,
            ), row=r, col=c)
    feat_fig.update_layout(template=TEMPLATE, height=400, margin=dict(l=50, r=20, t=40, b=40))

    # 循环滑块
    curves = load_curves(battery_id)
    n_curves = len(curves.get("voltage_curve", []))
    max_idx = max(n_curves - 1, 0)
    slider_marks = {}
    if n_curves > 0:
        step = max(1, n_curves // 10)
        for i in range(0, n_curves, step):
            slider_marks[i] = str(i)
        slider_marks[max_idx] = str(max_idx)

    return kpis, gauge_fig, cap_fig, feat_fig, max_idx, [0, min(5, max_idx)], slider_marks


@callback(
    Output("detail-voltage-curves", "figure"),
    [Input("detail-battery-select", "value"),
     Input("detail-cycle-slider", "value")],
)
def update_voltage_curves(battery_id: str, cycle_range: list[int]) -> go.Figure:
    curves = load_curves(battery_id)
    v_curves = curves.get("voltage_curve", [])
    fig = go.Figure()
    if not v_curves:
        fig.update_layout(template=TEMPLATE, height=350)
        return fig

    start, end = cycle_range
    indices = list(range(start, min(end + 1, len(v_curves))))
    n_show = len(indices)

    for j, idx in enumerate(indices):
        v = v_curves[idx]
        t_frac = j / max(n_show - 1, 1)
        r_val = int(96 + t_frac * 143)
        g_val = int(165 - t_frac * 97)
        b_val = int(250 - t_frac * 182)
        color = f"rgb({r_val},{g_val},{b_val})"
        fig.add_trace(go.Scatter(
            y=v, mode="lines", name=f"循环 {idx}",
            line=dict(color=color, width=1.5),
            opacity=0.7 + 0.3 * t_frac,
        ))

    fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="采样点", yaxis_title="电压 (V)",
        showlegend=True if n_show <= 15 else False,
        margin=dict(l=50, r=20, t=10, b=40),
    )
    return fig


def _kpi(label: str, value: str, delta: str, css: str) -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value", style={"fontSize": "1.5rem"}),
        html.Div(delta, className="kpi-delta") if delta else html.Div(),
    ], className=f"kpi-card {css}")
