"""锂电池数字孪生可视化平台 — 主入口。

启动: python -m src.app.app
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dash
from dash import html, dcc, callback, Input, Output

from src.app.pages import overview, battery_detail, digital_twin, prediction_arena, explainability, degradation_3d

# ============================================================
# 初始化
# ============================================================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="锂电池数字孪生平台",
    update_title="加载中...",
    assets_folder=str(Path(__file__).parent / "assets"),
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
)
server = app.server

# ============================================================
# 导航配置
# ============================================================
NAV_ITEMS = [
    {"path": "/", "label": "总览仪表盘", "icon": "bi-speedometer2"},
    {"path": "/detail", "label": "电池详情", "icon": "bi-battery-charging"},
    {"path": "/twin", "label": "数字孪生仿真", "icon": "bi-cpu"},
    {"path": "/arena", "label": "模型预测竞技场", "icon": "bi-trophy"},
    {"path": "/explain", "label": "可解释性分析", "icon": "bi-diagram-3"},
    {"path": "/3d", "label": "3D 退化景观", "icon": "bi-box"},
]


def make_sidebar() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("锂电池数字孪生"),
            html.P("大数据可视化平台"),
        ], className="sidebar-brand"),

        html.Nav([
            dcc.Link(
                html.Div([
                    html.I(className=f"bi {item['icon']}", style={"fontSize": "1.1rem"}),
                    html.Span(item["label"]),
                ], style={"display": "flex", "alignItems": "center", "gap": "0.75rem"}),
                href=item["path"],
                className="nav-link",
                id=f"nav-{item['path'].strip('/')}",
            )
            for item in NAV_ITEMS
        ]),
    ], className="sidebar")


# ============================================================
# 布局
# ============================================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    make_sidebar(),
    html.Div(id="page-content", className="main-content"),
])


# ============================================================
# 路由
# ============================================================
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def route_page(pathname: str) -> html.Div:
    if pathname == "/" or pathname is None:
        return overview.layout()
    if pathname == "/detail":
        return battery_detail.layout()
    if pathname == "/twin":
        return digital_twin.layout()
    if pathname == "/arena":
        return prediction_arena.layout()
    if pathname == "/explain":
        return explainability.layout()
    if pathname == "/3d":
        return degradation_3d.layout()

    return html.Div([
        html.H1("404 — 页面未找到"),
        html.P(f"路径 '{pathname}' 不存在"),
    ], className="page-header")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
