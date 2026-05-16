# 锂电池数字孪生与实时仿真可视化平台

> 华侨大学工学院 · 大数据可视化技术课程项目

基于 NASA PCoE 和 CALCE 锂电池退化数据集，构建集数据处理、多模型预测、数字孪生仿真与交互式可视化于一体的综合平台。

## 功能页面

| 页面 | 功能 |
|------|------|
| 车队总览 | KPI 卡片、健康矩阵热力图、容量退化曲线、模型雷达图 |
| 电池详情 | 单体深度分析、SOH 仪表盘、放电电压曲线对比 |
| 数字孪生仿真 | 1-RC Thevenin ECM 交互式放电仿真器 |
| 模型预测竞技场 | 8 种模型自由多选对比、置信区间、残差分析 |
| 可解释性分析 | SHAP 特征重要性、回归系数、跨电池热力图 |
| 3D 退化景观 | 三维散点图、多种着色策略、2D 投影 |

## 模型体系

**基础模型**: Linear Regression (Ridge), Random Forest, Transformer, PINN

**集成策略**: 加权集成, 堆叠元学习, 生命周期自适应, 物理约束集成

最优模型（加权集成）平均 R² = 0.9939

## 技术栈

- **可视化**: Dash 4.x + Plotly.js + CSS Custom Properties (深色主题)
- **建模**: PyTorch (Transformer, PINN) + scikit-learn (Ridge, RF)
- **数据**: Pandas + Parquet + NumPy
- **数字孪生**: 1-RC Thevenin ECM + 简化热模型

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 数据预处理（需要原始数据在 data/NASA/ 和 data/CALCE/ 下）
python scripts/preprocess.py

# 训练模型
python scripts/train_models.py

# 启动可视化平台
python -m src.app.app
# 访问 http://127.0.0.1:8050
```

## 项目结构

```
├── data/
│   └── processed/          # 处理后的数据（Parquet, JSON, 模型权重）
├── scripts/
│   ├── preprocess.py       # 数据预处理流水线
│   ├── train_models.py     # 模型训练脚本
│   └── generate_hqu_report.py  # 课程报告生成
├── src/
│   ├── app/                # Dash 可视化应用
│   │   ├── pages/          # 6 个功能页面
│   │   ├── assets/         # CSS 样式
│   │   ├── app.py          # 主入口
│   │   ├── chart_theme.py  # 图表主题
│   │   └── data_loader.py  # 数据加载层
│   ├── data/               # 数据解析与特征工程
│   ├── models/             # 模型实现
│   └── utils/              # 工具常量
└── requirements.txt
```

## 数据说明

原始数据（NASA .mat 文件和 CALCE .xlsx 文件）体积较大，未包含在仓库中。处理后的数据已包含在 `data/processed/` 目录下，可直接启动可视化平台。

如需从头处理，请自行下载：
- NASA: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
- CALCE: https://calce.umd.edu/battery-data
