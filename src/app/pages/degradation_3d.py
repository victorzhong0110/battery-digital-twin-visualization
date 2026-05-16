"""第6页：3D 退化景观。

多维电池老化交互式 3D 可视化：
容量 × 电阻 × 循环 → 按 SOH 着色。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import load_features, get_battery_ids
from src.app.chart_theme import (
    TEMPLATE, CHART_COLORS, HEALTH_COLORS, soh_color,
    BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
)


def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H1("3D 退化景观"),
            html.P("多维电池老化模式的交互式 3D 空间探索"),
        ], className="page-header"),

        # 控制面板
        html.Div([
            html.Div([
                html.Label("着色依据", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="3d-color-by",
                    options=[
                        {"label": "SOH（健康状态）", "value": "soh"},
                        {"label": "数据来源", "value": "source"},
                        {"label": "电池编号", "value": "battery_id"},
                        {"label": "容量衰减率", "value": "capacity_fade_rate"},
                    ],
                    value="soh",
                    clearable=False,
                    style={"width": "200px"},
                ),
            ]),
            html.Div([
                html.Label("Z 轴", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="3d-z-axis",
                    options=[
                        {"label": "容量 (Ah)", "value": "capacity_ah"},
                        {"label": "SOH", "value": "soh"},
                        {"label": "放电时长 (s)", "value": "discharge_duration_s"},
                        {"label": "电压斜率", "value": "voltage_slope"},
                    ],
                    value="capacity_ah",
                    clearable=False,
                    style={"width": "220px"},
                ),
            ]),
            html.Div([
                html.Label("点大小", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Slider(id="3d-point-size", min=2, max=8, step=1, value=4,
                           marks={2: "小", 4: "中", 6: "大", 8: "特大"}),
            ], style={"width": "180px"}),
        ], className="card-dark", style={"display": "flex", "gap": "1.5rem",
                                         "alignItems": "flex-end", "marginBottom": "1.5rem"}),

        # 3D 散点图
        html.Div([
            html.Div("退化景观（3D 散点）", className="chart-title"),
            dcc.Graph(id="3d-scatter", style={"height": "600px"}),
        ], className="chart-container"),

        # 2D 投影
        html.Div([
            html.Div([
                html.Div("容量 vs. 电阻（2D 投影）", className="chart-title"),
                dcc.Graph(id="3d-projection-cr"),
            ], className="chart-container", style={"flex": "1"}),
            html.Div([
                html.Div("SOH vs. 循环次数（2D 投影）", className="chart-title"),
                dcc.Graph(id="3d-projection-sc"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),
    ])


@callback(
    [Output("3d-scatter", "figure"),
     Output("3d-projection-cr", "figure"),
     Output("3d-projection-sc", "figure")],
    [Input("3d-color-by", "value"),
     Input("3d-z-axis", "value"),
     Input("3d-point-size", "value")],
)
def update_3d(color_by: str, z_axis: str, point_size: int):
    feat = load_features()

    # ---- 3D 散点图 ----
    z_label_map = {
        "capacity_ah": "容量 (Ah)",
        "soh": "SOH",
        "discharge_duration_s": "放电时长 (s)",
        "voltage_slope": "电压斜率",
    }
    z_label = z_label_map.get(z_axis, z_axis)
    fig_3d = go.Figure()

    if color_by == "soh":
        fig_3d.add_trace(go.Scatter3d(
            x=feat["cycle_index"],
            y=feat["internal_resistance_ohm"],
            z=feat[z_axis],
            mode="markers",
            marker=dict(
                size=point_size,
                color=feat["soh"],
                colorscale=[
                    [0, HEALTH_COLORS["critical"]],
                    [0.3, HEALTH_COLORS["warning"]],
                    [0.7, HEALTH_COLORS["good"]],
                    [1, HEALTH_COLORS["excellent"]],
                ],
                cmin=0.6, cmax=1.0,
                colorbar=dict(title="SOH", tickformat=".0%", len=0.6),
                opacity=0.8,
            ),
            text=feat["battery_id"],
            hovertemplate=(
                "电池: %{text}<br>"
                "循环: %{x}<br>"
                "电阻: %{y:.4f} Ω<br>"
                f"{z_label}: %{{z:.4f}}<br>"
                "SOH: %{{marker.color:.1%}}<extra></extra>"
            ),
        ))
    elif color_by == "source":
        for source, color in [("NASA", CHART_COLORS[0]), ("CALCE", CHART_COLORS[3])]:
            sdf = feat[feat["source"] == source]
            fig_3d.add_trace(go.Scatter3d(
                x=sdf["cycle_index"],
                y=sdf["internal_resistance_ohm"],
                z=sdf[z_axis],
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.8),
                name=source,
                text=sdf["battery_id"],
            ))
    elif color_by == "battery_id":
        for i, bid in enumerate(sorted(feat["battery_id"].unique())):
            bdf = feat[feat["battery_id"] == bid]
            fig_3d.add_trace(go.Scatter3d(
                x=bdf["cycle_index"],
                y=bdf["internal_resistance_ohm"],
                z=bdf[z_axis],
                mode="markers",
                marker=dict(size=point_size, color=CHART_COLORS[i % len(CHART_COLORS)], opacity=0.8),
                name=bid,
            ))
    else:
        fig_3d.add_trace(go.Scatter3d(
            x=feat["cycle_index"],
            y=feat["internal_resistance_ohm"],
            z=feat[z_axis],
            mode="markers",
            marker=dict(
                size=point_size,
                color=feat[color_by] if color_by in feat.columns else feat["soh"],
                colorscale="Viridis",
                colorbar=dict(title=color_by),
                opacity=0.8,
            ),
            text=feat["battery_id"],
        ))

    fig_3d.update_layout(
        template=TEMPLATE,
        height=600,
        scene=dict(
            xaxis=dict(title="循环次数", backgroundcolor=BG_CARD, gridcolor="rgba(42,53,72,0.3)"),
            yaxis=dict(title="电阻 (Ω)", backgroundcolor=BG_CARD, gridcolor="rgba(42,53,72,0.3)"),
            zaxis=dict(title=z_label, backgroundcolor=BG_CARD, gridcolor="rgba(42,53,72,0.3)"),
            bgcolor=BG_CARD,
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )

    # ---- 2D 投影：容量 vs 电阻 ----
    proj_cr = go.Figure()
    for i, bid in enumerate(sorted(feat["battery_id"].unique())):
        bdf = feat[feat["battery_id"] == bid].sort_values("cycle_index")
        proj_cr.add_trace(go.Scatter(
            x=bdf["internal_resistance_ohm"],
            y=bdf["capacity_ah"],
            mode="lines+markers",
            name=bid,
            marker=dict(color=CHART_COLORS[i % len(CHART_COLORS)], size=4),
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
        ))
    proj_cr.update_layout(
        template=TEMPLATE, height=380,
        xaxis_title="内阻 (Ω)",
        yaxis_title="容量 (Ah)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # ---- 2D 投影：SOH vs 循环 ----
    proj_sc = go.Figure()
    for i, bid in enumerate(sorted(feat["battery_id"].unique())):
        bdf = feat[feat["battery_id"] == bid].sort_values("cycle_index")
        proj_sc.add_trace(go.Scatter(
            x=bdf["cycle_index"],
            y=bdf["soh"],
            mode="lines",
            name=bid,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))
    proj_sc.add_hline(y=0.80, line_dash="dash", line_color=HEALTH_COLORS["warning"])
    proj_sc.update_layout(
        template=TEMPLATE, height=380,
        xaxis_title="循环次数",
        yaxis_title="SOH",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig_3d, proj_cr, proj_sc
