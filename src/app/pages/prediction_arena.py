"""第4页：模型预测竞技场。

8 种模型 + 4 种集成策略并排对比。
预测叠加、误差分布、置信区间、PINN 损失分解、Transformer 注意力热图。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

from src.app.data_loader import (
    load_predictions, load_metrics, get_battery_ids,
    load_pinn_loss_history, load_transformer_attention,
    load_ensemble_weights,
)
from src.app.chart_theme import (
    TEMPLATE, MODEL_COLORS, MODEL_LABELS, CHART_COLORS,
    HEALTH_COLORS, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
)

ALL_MODELS = [
    "linear", "rf", "transformer", "pinn",
    "ens_weighted", "ens_stacking", "ens_lifecycle", "ens_physics_constrained",
]

PRED_COLS = {
    "linear": "pred_linear",
    "rf": "pred_rf",
    "transformer": "pred_transformer",
    "pinn": "pred_pinn",
    "ens_weighted": "pred_ens_weighted",
    "ens_stacking": "pred_ens_stacking",
    "ens_lifecycle": "pred_ens_lifecycle",
    "ens_physics_constrained": "pred_ens_physics_constrained",
}


def layout() -> html.Div:
    battery_ids = get_battery_ids()
    model_options = [{"label": MODEL_LABELS.get(m, m), "value": m} for m in ALL_MODELS]

    return html.Div([
        html.Div([
            html.H1("模型预测竞技场"),
            html.P("8 种模型 × 4 种集成策略的全维度对比与不确定性可视化"),
        ], className="page-header"),

        html.Div([
            html.Div([
                html.Label("电池", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="arena-battery",
                    options=[{"label": b, "value": b} for b in battery_ids],
                    value=battery_ids[0] if battery_ids else None,
                    clearable=False, style={"width": "200px"},
                ),
            ]),
            html.Div([
                html.Label("对比模型", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="arena-models", options=model_options,
                    value=["linear", "rf", "pinn", "ens_weighted"],
                    multi=True, style={"width": "500px"},
                ),
            ]),
            html.Div([
                html.Label("显示选项", style={"color": TEXT_SECONDARY, "fontSize": "0.8rem"}),
                dcc.Checklist(
                    id="arena-show-ci",
                    options=[{"label": " 置信区间", "value": "show"}],
                    value=["show"], inline=True,
                ),
            ]),
        ], className="card-dark", style={"display": "flex", "gap": "1.5rem",
                                         "flexWrap": "wrap", "alignItems": "flex-end",
                                         "marginBottom": "1.5rem"}),

        html.Div(id="arena-metrics-table", className="card-dark", style={"marginBottom": "1.5rem"}),

        html.Div([
            html.Div("预测值 vs. 真实值", className="chart-title"),
            dcc.Graph(id="arena-predictions"),
        ], className="chart-container"),

        html.Div([
            html.Div([
                html.Div("误差分布", className="chart-title"),
                dcc.Graph(id="arena-error-dist"),
            ], className="chart-container", style={"flex": "1"}),
            html.Div([
                html.Div("残差随循环变化", className="chart-title"),
                dcc.Graph(id="arena-residuals"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        html.Div([
            html.Div([
                html.Div("PINN 损失分解", className="chart-title"),
                dcc.Graph(id="arena-pinn-loss"),
            ], className="chart-container", style={"flex": "1"}),
            html.Div([
                html.Div("集成策略权重分布", className="chart-title"),
                dcc.Graph(id="arena-ensemble-weights"),
            ], className="chart-container", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem"}),

        html.Div([
            html.Div("Transformer 注意力热图", className="chart-title"),
            dcc.Graph(id="arena-attention"),
        ], className="chart-container"),
    ])


@callback(
    [Output("arena-metrics-table", "children"),
     Output("arena-predictions", "figure"),
     Output("arena-error-dist", "figure"),
     Output("arena-residuals", "figure"),
     Output("arena-pinn-loss", "figure"),
     Output("arena-ensemble-weights", "figure"),
     Output("arena-attention", "figure")],
    [Input("arena-battery", "value"),
     Input("arena-models", "value"),
     Input("arena-show-ci", "value")],
)
def update_arena(battery_id: str, selected_models: list[str], show_ci: list[str]):
    preds_df = load_predictions()
    metrics = load_metrics()
    bdf = preds_df[preds_df["battery_id"] == battery_id].sort_values("cycle_index")

    if bdf.empty or not selected_models:
        empty = go.Figure().update_layout(template=TEMPLATE, height=350)
        return html.Div(), empty, empty, empty, empty, empty, empty

    actual = bdf["capacity_actual"].values
    cycles = bdf["cycle_index"].values
    show_ci_flag = "show" in (show_ci or [])

    # ---- 指标表格 ----
    table_rows = []
    for m in selected_models:
        key = f"{m}_{battery_id}"
        if key in metrics:
            met = metrics[key]
            table_rows.append(html.Tr([
                html.Td(MODEL_LABELS.get(m, m), style={"color": MODEL_COLORS.get(m, "#fff"), "fontWeight": "600"}),
                html.Td(f"{met.get('rmse', 0):.4f}"),
                html.Td(f"{met.get('mae', 0):.4f}"),
                html.Td(f"{met.get('r2', 0):.4f}"),
                html.Td(f"{met.get('mape', 0):.2f}%"),
            ]))

    metrics_table = html.Table([
        html.Thead(html.Tr([
            html.Th("模型"), html.Th("RMSE"), html.Th("MAE"), html.Th("R²"), html.Th("MAPE"),
        ])),
        html.Tbody(table_rows),
    ], style={"width": "100%", "color": TEXT_PRIMARY, "borderCollapse": "collapse", "fontSize": "0.9rem"})

    # ---- 预测图 ----
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=cycles, y=actual, mode="lines+markers", name="真实值",
        line=dict(color=MODEL_COLORS["actual"], width=2.5), marker=dict(size=4),
    ))

    for m in selected_models:
        col = PRED_COLS.get(m)
        if col and col in bdf.columns:
            y_pred = bdf[col].values
            color = MODEL_COLORS.get(m, "#888")
            pred_fig.add_trace(go.Scatter(
                x=cycles, y=y_pred, mode="lines",
                name=MODEL_LABELS.get(m, m), line=dict(color=color, width=2),
            ))
            if show_ci_flag:
                lower_col, upper_col = f"{col}_lower", f"{col}_upper"
                if lower_col in bdf.columns and upper_col in bdf.columns:
                    lb, ub = bdf[lower_col].values, bdf[upper_col].values
                    valid = ~np.isnan(lb) & ~np.isnan(ub)
                    if valid.any():
                        pred_fig.add_trace(go.Scatter(
                            x=np.concatenate([cycles[valid], cycles[valid][::-1]]),
                            y=np.concatenate([ub[valid], lb[valid][::-1]]),
                            fill="toself",
                            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba")
                                           if "rgb" in color else f"rgba(100,100,100,0.1)",
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))

    pred_fig.update_layout(
        template=TEMPLATE, height=420,
        xaxis_title="循环次数", yaxis_title="容量 (Ah)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # ---- 误差分布 ----
    err_fig = go.Figure()
    for m in selected_models:
        col = PRED_COLS.get(m)
        if col and col in bdf.columns:
            errors = actual - bdf[col].values
            valid = ~np.isnan(errors)
            if valid.any():
                err_fig.add_trace(go.Histogram(
                    x=errors[valid], name=MODEL_LABELS.get(m, m),
                    marker_color=MODEL_COLORS.get(m, "#888"), opacity=0.7, nbinsx=20,
                ))
    err_fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="误差 (真实 - 预测)", yaxis_title="计数", barmode="overlay",
    )

    # ---- 残差 ----
    res_fig = go.Figure()
    for m in selected_models:
        col = PRED_COLS.get(m)
        if col and col in bdf.columns:
            residuals = actual - bdf[col].values
            valid = ~np.isnan(residuals)
            if valid.any():
                res_fig.add_trace(go.Scatter(
                    x=cycles[valid], y=residuals[valid], mode="markers",
                    name=MODEL_LABELS.get(m, m),
                    marker=dict(color=MODEL_COLORS.get(m, "#888"), size=5, opacity=0.7),
                ))
    res_fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY)
    res_fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="循环次数", yaxis_title="残差 (Ah)",
    )

    # ---- PINN 损失分解 ----
    pinn_history = load_pinn_loss_history(battery_id)
    pinn_fig = go.Figure()
    if pinn_history:
        epochs = list(range(len(pinn_history)))
        components = ["data", "physics", "monotone", "boundary"]
        comp_labels = ["数据损失", "物理损失", "单调性损失", "边界损失"]
        comp_colors = [CHART_COLORS[0], CHART_COLORS[3], CHART_COLORS[2], CHART_COLORS[4]]
        for comp, label, color in zip(components, comp_labels, comp_colors):
            vals = [h.get(comp, 0) for h in pinn_history]
            pinn_fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode="lines", name=label,
                line=dict(color=color, width=2), stackgroup="one",
            ))
    pinn_fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="训练轮次", yaxis_title="损失值", yaxis_type="log",
    )

    # ---- 集成权重 ----
    ens_fig = go.Figure()
    strategies = ["weighted", "stacking", "lifecycle", "physics_constrained"]
    base_models = ["linear", "rf", "transformer", "pinn"]
    base_labels = ["线性回归", "随机森林", "Transformer", "PINN"]
    for strat in strategies:
        weights = load_ensemble_weights(strat, battery_id)
        if weights is not None and len(weights) > 0:
            avg_w = weights.mean(axis=0)
            ens_fig.add_trace(go.Bar(
                x=base_labels, y=avg_w[:len(base_models)],
                name=MODEL_LABELS.get(f"ens_{strat}", strat),
                marker_color=MODEL_COLORS.get(f"ens_{strat}", "#888"),
            ))
    ens_fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="基础模型", yaxis_title="平均权重", barmode="group",
    )

    # ---- Transformer 注意力 ----
    attn = load_transformer_attention(battery_id)
    attn_fig = go.Figure()
    if attn is not None and len(attn) > 0:
        n_show = min(len(attn), 50)
        step = max(1, len(attn) // n_show)
        attn_sample = attn[::step]
        attn_fig.add_trace(go.Heatmap(
            z=attn_sample.T,
            x=[f"预测 {i * step}" for i in range(len(attn_sample))],
            y=[f"t-{i}" for i in range(attn_sample.shape[1] - 1, -1, -1)],
            colorscale=[[0, BG_CARD], [1, CHART_COLORS[0]]],
            colorbar=dict(title="注意力"),
            hovertemplate="预测点: %{x}<br>历史偏移: %{y}<br>权重: %{z:.3f}<extra></extra>",
        ))
    attn_fig.update_layout(
        template=TEMPLATE, height=350,
        xaxis_title="预测索引", yaxis_title="历史循环偏移",
    )

    return metrics_table, pred_fig, err_fig, res_fig, pinn_fig, ens_fig, attn_fig
