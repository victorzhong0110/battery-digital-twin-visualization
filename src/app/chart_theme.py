"""Plotly 图表主题 — 暗色奢华风格。

提供统一的模板和配色方案。
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# 配色方案
CHART_COLORS = [
    "#60a5fa",  # 蓝
    "#a78bfa",  # 紫
    "#34d399",  # 绿
    "#fbbf24",  # 琥珀
    "#f472b6",  # 粉
    "#38bdf8",  # 天蓝
    "#fb923c",  # 橙
    "#c084fc",  # 紫罗兰
]

HEALTH_COLORS = {
    "excellent": "#10b981",
    "good": "#34d399",
    "warning": "#f59e0b",
    "critical": "#ef4444",
    "dead": "#6b7280",
}

MODEL_COLORS = {
    "linear": "#60a5fa",
    "rf": "#34d399",
    "transformer": "#a78bfa",
    "pinn": "#fbbf24",
    "ens_weighted": "#f472b6",
    "ens_stacking": "#38bdf8",
    "ens_lifecycle": "#fb923c",
    "ens_physics_constrained": "#c084fc",
    "actual": "#f0f4f8",
}

MODEL_LABELS = {
    "linear": "线性回归 (Ridge)",
    "rf": "随机森林",
    "transformer": "Transformer",
    "pinn": "PINN (物理约束)",
    "ens_weighted": "加权集成",
    "ens_stacking": "堆叠元学习",
    "ens_lifecycle": "生命周期自适应",
    "ens_physics_constrained": "物理约束集成",
}

BG_PRIMARY = "#0a0e17"
BG_CARD = "#1a2332"
GRID_COLOR = "rgba(42, 53, 72, 0.5)"
TEXT_PRIMARY = "#f0f4f8"
TEXT_SECONDARY = "#94a3b8"


def get_template() -> go.layout.Template:
    """构建暗色奢华 Plotly 模板。"""
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=BG_CARD,
            plot_bgcolor=BG_CARD,
            font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_PRIMARY, size=12),
            title=dict(font=dict(size=16, color=TEXT_PRIMARY)),
            xaxis=dict(
                gridcolor=GRID_COLOR,
                zerolinecolor=GRID_COLOR,
                title_font=dict(color=TEXT_SECONDARY, size=12),
                tickfont=dict(color=TEXT_SECONDARY, size=11),
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR,
                zerolinecolor=GRID_COLOR,
                title_font=dict(color=TEXT_SECONDARY, size=12),
                tickfont=dict(color=TEXT_SECONDARY, size=11),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_SECONDARY, size=11),
            ),
            margin=dict(l=50, r=20, t=40, b=40),
            colorway=CHART_COLORS,
        )
    )


TEMPLATE = get_template()


def styled_fig(fig: go.Figure) -> go.Figure:
    """为图表应用暗色奢华主题。"""
    fig.update_layout(template=TEMPLATE)
    return fig


def soh_color(soh: float) -> str:
    """SOH 值映射到健康色。"""
    if soh >= 0.95:
        return HEALTH_COLORS["excellent"]
    if soh >= 0.80:
        return HEALTH_COLORS["good"]
    if soh >= 0.70:
        return HEALTH_COLORS["warning"]
    if soh >= 0.50:
        return HEALTH_COLORS["critical"]
    return HEALTH_COLORS["dead"]


def soh_label(soh: float) -> str:
    """SOH 值映射到健康标签。"""
    if soh >= 0.95:
        return "优秀"
    if soh >= 0.80:
        return "良好"
    if soh >= 0.70:
        return "警告"
    if soh >= 0.50:
        return "危险"
    return "寿命终止"
