"""第5页：模型可解释性分析。

SHAP 特征重要性、线性系数、跨电池特征对比、数据效率分析。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import load_explanations, load_metrics, get_battery_ids
from src.app.chart_theme import (
    TEMPLATE, CHART_COLORS, MODEL_COLORS,
    BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
)


def layout() -> html.Div:
    battery_ids = get_battery_ids()
    return html.Div([
        html.Div([
            html.H1("可解释性分析"),
            html.P("SHAP 特征重要性、线性系数、跨模型洞察"),
        ], className="page-header"),

        html.Div([
            html.Div([
                html.Label("电池", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="explain-battery",
                    options=[{"label": b, "value": b} for b in battery_ids],
                    value=battery_ids[0] if battery_ids else None,
                    clearable=False, style={"width": "220px"},
                ),
            ]),
        ], className="card-dark", style={"display": "flex", "gap": "1.5rem", "marginBottom": "1.5rem"}),

        html.Div([
            html.Div([
                html.Div("随机森林 — 特征重要性 (SHAP)", className="chart-title"),
                dcc.Graph(id="explain-rf-importance"),
            ], className="chart-container", style={"flex": "1"}),
            html.Div([
                html.Div("线性模型 — 回归系数", className="chart-title"),
                dcc.Graph(id="explain-linear-coefs"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        html.Div([
            html.Div("跨电池特征重要性热力图", className="chart-title"),
            dcc.Graph(id="explain-cross-battery"),
        ], className="chart-container"),

        html.Div([
            html.Div("模型 R² vs. 训练数据量（数据效率分析）", className="chart-title"),
            dcc.Graph(id="explain-data-efficiency"),
        ], className="chart-container"),
    ])


@callback(
    [Output("explain-rf-importance", "figure"),
     Output("explain-linear-coefs", "figure"),
     Output("explain-cross-battery", "figure"),
     Output("explain-data-efficiency", "figure")],
    Input("explain-battery", "value"),
)
def update_explain(battery_id: str):
    expl = load_explanations()
    metrics = load_metrics()
    empty = go.Figure().update_layout(template=TEMPLATE, height=350)

    # ---- RF 特征重要性 ----
    rf_key = f"rf_{battery_id}"
    rf_fig = go.Figure()
    if rf_key in expl:
        rf_data = expl[rf_key]
        names = rf_data["feature_names"]
        importance = np.array(rf_data["feature_importance"])
        sorted_idx = np.argsort(importance)
        sorted_names = [names[i] for i in sorted_idx]
        sorted_imp = importance[sorted_idx]
        n = len(sorted_names)
        colors = [f"rgba(96, 165, 250, {0.3 + 0.7 * i / n})" for i in range(n)]

        rf_fig.add_trace(go.Bar(
            x=sorted_imp, y=sorted_names, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        rf_fig.update_layout(
            template=TEMPLATE, height=400,
            xaxis_title="平均 |SHAP 值|",
            margin=dict(l=180, r=20, t=10, b=40),
        )
    else:
        rf_fig = empty

    # ---- 线性系数 ----
    lr_key = f"linear_{battery_id}"
    lr_fig = go.Figure()
    if lr_key in expl:
        lr_data = expl[lr_key]
        names = lr_data["feature_names"]
        coefs = lr_data.get("coefficients", lr_data["feature_importance"])
        sorted_idx = np.argsort(np.abs(coefs))
        sorted_names = [names[i] for i in sorted_idx]
        sorted_coefs = [coefs[i] for i in sorted_idx]
        colors = [CHART_COLORS[2] if c >= 0 else CHART_COLORS[4] for c in sorted_coefs]

        lr_fig.add_trace(go.Bar(
            x=sorted_coefs, y=sorted_names, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        lr_fig.update_layout(
            template=TEMPLATE, height=400,
            xaxis_title="系数（按特征标准差缩放）",
            margin=dict(l=180, r=20, t=10, b=40),
        )
    else:
        lr_fig = empty

    # ---- 跨电池热力图 ----
    cross_fig = go.Figure()
    all_batteries = sorted(set(k.replace("rf_", "") for k in expl if k.startswith("rf_")))
    if all_batteries:
        feature_names = expl[f"rf_{all_batteries[0]}"]["feature_names"]
        z_matrix = []
        for bid in all_batteries:
            key = f"rf_{bid}"
            if key in expl:
                imp_arr = np.array(expl[key]["feature_importance"])
                imp_max = imp_arr.max()
                if imp_max > 0:
                    imp_arr = imp_arr / imp_max
                z_matrix.append(imp_arr.tolist())
            else:
                z_matrix.append([0] * len(feature_names))

        cross_fig.add_trace(go.Heatmap(
            z=z_matrix, x=feature_names, y=all_batteries,
            colorscale=[[0, BG_CARD], [0.5, "#1e40af"], [1, "#60a5fa"]],
            colorbar=dict(title="归一化"),
            hovertemplate="电池: %{y}<br>特征: %{x}<br>重要性: %{z:.3f}<extra></extra>",
        ))
        cross_fig.update_layout(
            template=TEMPLATE, height=350,
            xaxis=dict(tickangle=45),
            margin=dict(l=120, r=20, t=10, b=100),
        )
    else:
        cross_fig = empty

    # ---- 数据效率 ----
    from src.app.data_loader import load_features
    feat = load_features()
    eff_fig = go.Figure()
    model_prefixes = ["linear", "rf", "transformer", "pinn"]
    model_zh = {"linear": "线性回归", "rf": "随机森林", "transformer": "Transformer", "pinn": "PINN"}

    for m in model_prefixes:
        x_sizes, y_r2 = [], []
        for bid in feat["battery_id"].unique():
            key = f"{m}_{bid}"
            if key in metrics:
                r2 = metrics[key].get("r2", np.nan)
                if not np.isnan(r2):
                    total = len(feat)
                    test_size = len(feat[feat["battery_id"] == bid])
                    x_sizes.append(total - test_size)
                    y_r2.append(r2)
        if x_sizes:
            eff_fig.add_trace(go.Scatter(
                x=x_sizes, y=y_r2, mode="markers+lines",
                name=model_zh.get(m, m),
                marker=dict(color=MODEL_COLORS.get(m, "#888"), size=8),
                line=dict(color=MODEL_COLORS.get(m, "#888"), width=1.5, dash="dot"),
            ))

    eff_fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)
    eff_fig.update_layout(
        template=TEMPLATE, height=380,
        xaxis_title="训练数据量（样本数）", yaxis_title="R²",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return rf_fig, lr_fig, cross_fig, eff_fig
