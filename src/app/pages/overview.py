"""第1页：车队总览仪表盘。

KPI 卡片 + 车队健康热力图 + SOH 分布 + 退化曲线 + 电阻趋势 + 模型雷达图。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import load_features, load_metrics, get_battery_summary
from src.app.chart_theme import (
    TEMPLATE, CHART_COLORS, soh_color, soh_label,
    HEALTH_COLORS, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR,
)


def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H1("车队总览"),
            html.P("实时电池车队健康监控与退化分析"),
        ], className="page-header"),

        html.Div(id="overview-kpi-grid", className="kpi-grid"),

        # 行1：健康热力图 + SOH 分布
        html.Div([
            html.Div([
                html.Div("车队健康矩阵", className="chart-title"),
                dcc.Graph(id="fleet-heatmap"),
            ], className="chart-container", style={"flex": "1.2"}),
            html.Div([
                html.Div("最终 SOH 分布", className="chart-title"),
                dcc.Graph(id="soh-distribution"),
            ], className="chart-container", style={"flex": "0.8"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        # 行2：容量退化 + 电阻趋势
        html.Div([
            html.Div([
                html.Div("容量退化曲线", className="chart-title"),
                dcc.Graph(id="capacity-curves"),
            ], className="chart-container", style={"flex": "1"}),
            html.Div([
                html.Div("内阻变化趋势", className="chart-title"),
                dcc.Graph(id="resistance-curves"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        # 行3：模型性能雷达图
        html.Div([
            html.Div("模型性能对比（平均 R²）", className="chart-title"),
            dcc.Graph(id="model-radar"),
        ], className="chart-container"),

        dcc.Store(id="overview-trigger", data=True),
    ])


@callback(
    Output("overview-kpi-grid", "children"),
    Input("overview-trigger", "data"),
)
def update_kpis(_: bool) -> list:
    summary = get_battery_summary()
    feat = load_features()

    avg_soh = summary["final_soh"].mean()
    worst_battery = summary.loc[summary["final_soh"].idxmin()]
    total_cycles = feat.shape[0]
    max_fade = summary["capacity_fade_pct"].max()

    health_class = "health-excellent" if avg_soh > 0.90 else (
        "health-warning" if avg_soh > 0.80 else "health-critical"
    )

    return [
        _kpi_card("车队平均 SOH", f"{avg_soh:.1%}", soh_label(avg_soh), health_class),
        _kpi_card("电池总数", str(len(summary)), f"共 {total_cycles} 个循环", ""),
        _kpi_card("最大容量衰减", f"{max_fade:.1f}%", worst_battery["battery_id"], "health-warning"),
        _kpi_card("最差电池 SOH", f"{worst_battery['final_soh']:.1%}",
                   worst_battery["battery_id"],
                   "health-critical" if worst_battery["final_soh"] < 0.80 else "health-warning"),
    ]


def _kpi_card(label: str, value: str, delta: str, css_class: str) -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(delta, className="kpi-delta"),
    ], className=f"kpi-card {css_class}")


@callback(
    Output("fleet-heatmap", "figure"),
    Input("overview-trigger", "data"),
)
def update_heatmap(_: bool) -> go.Figure:
    feat = load_features()
    batteries = sorted(feat["battery_id"].unique())

    max_cycles = feat.groupby("battery_id")["cycle_index"].max().max()
    n_bins = 20
    bin_edges = np.linspace(0, max_cycles, n_bins + 1)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(n_bins)]

    z_matrix = []
    for bid in batteries:
        bdf = feat[feat["battery_id"] == bid]
        row = []
        for i in range(n_bins):
            mask = (bdf["cycle_index"] >= bin_edges[i]) & (bdf["cycle_index"] < bin_edges[i + 1])
            if mask.any():
                row.append(float(bdf.loc[mask, "soh"].mean()))
            else:
                row.append(np.nan)
        z_matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=z_matrix, x=bin_labels, y=batteries,
        colorscale=[
            [0, HEALTH_COLORS["critical"]], [0.3, HEALTH_COLORS["warning"]],
            [0.7, HEALTH_COLORS["good"]], [1, HEALTH_COLORS["excellent"]],
        ],
        zmin=0.6, zmax=1.0,
        colorbar=dict(title="SOH", tickformat=".0%"),
        hoverongaps=False,
        hovertemplate="电池: %{y}<br>循环范围: %{x}<br>SOH: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, height=320,
        xaxis_title="循环范围", yaxis_title="",
        margin=dict(l=100, r=20, t=10, b=40),
    )
    return fig


@callback(
    Output("soh-distribution", "figure"),
    Input("overview-trigger", "data"),
)
def update_soh_dist(_: bool) -> go.Figure:
    summary = get_battery_summary()
    colors = [soh_color(s) for s in summary["final_soh"]]

    fig = go.Figure(go.Bar(
        x=summary["battery_id"], y=summary["final_soh"],
        marker_color=colors,
        text=[f"{s:.1%}" for s in summary["final_soh"]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="电池: %{x}<br>最终 SOH: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.80, line_dash="dash", line_color=HEALTH_COLORS["warning"],
                  annotation_text="80% 警告线", annotation_font_color=HEALTH_COLORS["warning"])
    fig.update_layout(
        template=TEMPLATE, height=320,
        yaxis_title="最终 SOH", yaxis_tickformat=".0%", yaxis_range=[0.5, 1.05],
        showlegend=False, margin=dict(l=50, r=20, t=10, b=40),
    )
    return fig


@callback(
    Output("capacity-curves", "figure"),
    Input("overview-trigger", "data"),
)
def update_capacity(_: bool) -> go.Figure:
    feat = load_features()
    fig = go.Figure()
    for i, bid in enumerate(sorted(feat["battery_id"].unique())):
        bdf = feat[feat["battery_id"] == bid].sort_values("cycle_index")
        fig.add_trace(go.Scatter(
            x=bdf["cycle_index"], y=bdf["capacity_ah"],
            mode="lines", name=bid,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"{bid}<br>循环: %{{x}}<br>容量: %{{y:.3f}} Ah<extra></extra>",
        ))
    fig.update_layout(
        template=TEMPLATE, height=380,
        xaxis_title="循环次数", yaxis_title="容量 (Ah)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=30, b=40),
    )
    return fig


@callback(
    Output("resistance-curves", "figure"),
    Input("overview-trigger", "data"),
)
def update_resistance(_: bool) -> go.Figure:
    feat = load_features()
    fig = go.Figure()
    for i, bid in enumerate(sorted(feat["battery_id"].unique())):
        bdf = feat[feat["battery_id"] == bid].sort_values("cycle_index")
        r = bdf["internal_resistance_ohm"]
        if r.notna().any():
            fig.add_trace(go.Scatter(
                x=bdf["cycle_index"], y=r, mode="lines", name=bid,
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            ))
    fig.update_layout(
        template=TEMPLATE, height=380,
        xaxis_title="循环次数", yaxis_title="内阻 (Ω)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    return fig


@callback(
    Output("model-radar", "figure"),
    Input("overview-trigger", "data"),
)
def update_radar(_: bool) -> go.Figure:
    metrics = load_metrics()
    model_prefixes = [
        "linear", "rf", "transformer", "pinn",
        "ens_weighted", "ens_stacking", "ens_lifecycle", "ens_physics_constrained",
    ]
    labels = [
        "线性回归", "随机森林", "Transformer", "PINN",
        "加权集成", "堆叠元学习", "生命周期自适应", "物理约束集成",
    ]
    colors = [
        "#60a5fa", "#34d399", "#a78bfa", "#fbbf24",
        "#f472b6", "#38bdf8", "#fb923c", "#c084fc",
    ]

    avg_r2 = []
    for prefix in model_prefixes:
        keys = [k for k in metrics if k.startswith(prefix + "_")]
        r2_vals = [metrics[k]["r2"] for k in keys if not np.isnan(metrics[k].get("r2", np.nan))]
        avg_r2.append(max(np.mean(r2_vals), 0) if r2_vals else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_r2, theta=labels, fill="toself",
        fillcolor="rgba(96, 165, 250, 0.15)",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=8, color=colors),
        hovertemplate="%{theta}<br>平均 R²: %{r:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, height=420,
        polar=dict(
            bgcolor=BG_CARD,
            radialaxis=dict(range=[0, 1], gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_SECONDARY, size=10)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_SECONDARY, size=11)),
        ),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig
