const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TableOfContents,
} = require("docx");

// ── Color & Style Constants ──
const BLUE = "2E5EAA";
const DARK = "1A1A2E";
const GRAY = "666666";
const LIGHT_BG = "EAF0F7";
const WHITE = "FFFFFF";

// ── Helpers ──
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, bold: true, size: 32, font: "SimHei", color: DARK })],
  });
}
function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, bold: true, size: 28, font: "SimHei", color: BLUE })],
  });
}
function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, bold: true, size: 24, font: "SimHei", color: DARK })],
  });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: opts.indent !== false ? { firstLine: 480 } : undefined,
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text,
      size: 24, // 12pt
      font: "SimSun",
      color: opts.color || "333333",
      bold: opts.bold || false,
      italics: opts.italics || false,
    })],
  });
}
function emptyLine() {
  return new Paragraph({ spacing: { after: 80 }, children: [] });
}

// ── Table builder ──
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const TABLE_W = 9360;

function makeTable(headers, rows, colWidths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: h, bold: true, size: 22, font: "SimHei", color: WHITE })],
        })],
      })
    ),
  });

  const dataRows = rows.map(
    (row) =>
      new TableRow({
        children: row.map((cell, i) =>
          new TableCell({
            borders,
            width: { size: colWidths[i], type: WidthType.DXA },
            margins: { top: 50, bottom: 50, left: 100, right: 100 },
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: String(cell), size: 21, font: "SimSun" })],
            })],
          })
        ),
      })
  );

  return new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ── Bullet list ──
const numberingConfig = [
  {
    reference: "bullets",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } },
    }],
  },
];

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 340 },
    children: [new TextRun({ text, size: 22, font: "SimSun" })],
  });
}

// ══════════════════════════════════════════════════════
// BUILD DOCUMENT
// ══════════════════════════════════════════════════════

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "SimSun", size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: { config: numberingConfig },
  sections: [
    // ═══ COVER PAGE ═══
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        emptyLine(), emptyLine(), emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "大数据可视化技术", size: 36, font: "SimHei", color: GRAY })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "课程项目报告", size: 32, font: "SimHei", color: GRAY })],
        }),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 8 } },
          children: [new TextRun({
            text: "锂电池数字孪生与实时仿真可视化平台",
            size: 48, bold: true, font: "SimHei", color: DARK,
          })],
        }),
        emptyLine(), emptyLine(), emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 120 },
          children: [new TextRun({ text: "基于 NASA & CALCE 锂电池数据集", size: 26, font: "SimSun", color: GRAY })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 120 },
          children: [new TextRun({ text: "Transformer + PINN + 集成策略 | Dash + Plotly 可视化", size: 26, font: "SimSun", color: GRAY })],
        }),
        emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 60 },
          children: [new TextRun({ text: "2025 年 5 月", size: 24, font: "SimSun", color: GRAY })],
        }),
      ],
    },

    // ═══ TABLE OF CONTENTS ═══
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "锂电池数字孪生可视化平台 | 课程报告", size: 18, font: "SimSun", color: GRAY, italics: true })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "- ", size: 18, color: GRAY }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GRAY }), new TextRun({ text: " -", size: 18, color: GRAY })],
          })],
        }),
      },
      children: [
        heading1("目录"),
        new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 1: 项目概述 ═══
        heading1("一、项目概述"),

        heading2("1.1 研究背景"),
        para("锂离子电池作为新能源汽车、储能系统和便携式电子设备的核心动力来源，其健康状态（State of Health, SOH）的准确评估对于保障运行安全和延长使用寿命具有重要意义。随着电池管理系统（BMS）采集数据量的爆发式增长，如何利用大数据可视化技术直观呈现电池退化规律、辅助运维决策，已成为学术界和工业界的共同关注点。"),
        para("本项目以 NASA PCoE 和 CALCE 两大公开锂电池退化数据集为基础，构建了一套集数据处理、多模型预测、数字孪生仿真与交互式可视化于一体的综合平台，旨在探索大数据可视化技术在电池健康管理领域的创新应用。"),

        heading2("1.2 项目目标"),
        bullet("整合 NASA 与 CALCE 异构电池数据集，建立统一的数据处理流水线"),
        bullet("实现从传统统计模型到深度学习的递进式模型体系（线性回归、随机森林、Transformer、PINN）"),
        bullet("设计 4 种集成策略融合多模型优势，提升预测鲁棒性"),
        bullet("构建基于等效电路模型（ECM）的数字孪生仿真器，支持实时参数调节"),
        bullet("开发 6 页交互式可视化仪表盘，覆盖车队监控、单体分析、预测对比、可解释性等维度"),

        heading2("1.3 技术栈"),
        makeTable(
          ["层次", "技术选型", "说明"],
          [
            ["数据处理", "Python / Pandas / NumPy", "异构数据解析、特征工程、统一模式"],
            ["机器学习", "scikit-learn / Ridge / RandomForest", "传统 ML 基线模型"],
            ["深度学习", "PyTorch / Transformer / PINN", "注意力机制 + 物理约束神经网络"],
            ["集成策略", "自研 Ensemble 模块", "加权 / 堆叠 / 生命周期自适应 / 物理约束"],
            ["数字孪生", "1-RC Thevenin ECM + ODE 求解", "等效电路参数标定与实时仿真"],
            ["可解释性", "SHAP / TreeExplainer", "特征重要性与线性系数分析"],
            ["可视化", "Dash 4.x + Plotly.js", "响应式深色主题仪表盘"],
            ["图标资源", "Bootstrap Icons", "专业矢量图标库"],
          ],
          [1800, 3200, 4360]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 2: 数据采集与处理 ═══
        heading1("二、数据采集与处理"),

        heading2("2.1 数据来源"),
        para("本项目使用两个权威公开数据集，覆盖不同实验条件与电池型号："),

        makeTable(
          ["数据集", "电池编号", "循环数范围", "测试条件"],
          [
            ["NASA PCoE", "B0005, B0006, B0007, B0018", "100~170 次", "室温恒流充放电至失效"],
            ["CALCE", "CS2_35, CS2_36, CS2_37, CS2_38", "52~100 次", "不同温度与倍率老化测试"],
          ],
          [1800, 3000, 1800, 2760]
        ),
        emptyLine(),
        para("共包含 8 块电池、846 个循环周期的完整退化数据，涵盖电压、电流、温度、容量等多维时序信号。"),

        heading2("2.2 数据统一与特征工程"),
        para("由于两个数据集的格式差异显著（NASA 使用 .mat 文件，CALCE 使用 Excel 表格），本项目首先设计了统一数据模式（Unified Schema），通过专用解析器将异构数据转换为标准 Parquet 格式。"),

        heading3("2.2.1 统一模式字段"),
        para("统一后的数据表包含 20 个字段，涵盖基础标识（battery_id、source、cycle_index）、核心容量指标（capacity_ah、soh、rated_capacity_ah）、电化学特征（internal_resistance_ohm、charge_transfer_resistance_ohm）、温度特征（max_temp_c、mean_temp_c、ambient_temp_c、temp_rise）、时序特征（discharge_duration_s、voltage_slope）以及衍生特征（capacity_fade_rate、resistance_increase_rate、capacity_rolling_std、resistance_rolling_mean、capacity_normalized、resistance_normalized）。", { indent: false }),

        heading3("2.2.2 特征工程策略"),
        bullet("滚动统计：窗口为 5 个循环的容量标准差与电阻均值，捕捉短期波动"),
        bullet("归一化处理：基于额定容量的容量归一化和基于初始电阻的电阻归一化"),
        bullet("退化速率：容量衰减率和电阻增长率的逐循环差分计算"),
        bullet("温度特征：充放电过程中的温升幅度，反映内部产热变化"),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 3: 模型设计与实现 ═══
        heading1("三、模型设计与实现"),
        para("本项目采用递进式模型架构，从传统统计方法逐步过渡到深度学习与物理约束模型，最终通过集成策略融合各模型优势。这种设计体现了「从简单到复杂、从数据驱动到物理融合」的技术路线。"),

        heading2("3.1 基础模型"),

        heading3("3.1.1 线性回归（Ridge）"),
        para("采用 L2 正则化的 Ridge 回归作为基线模型。尽管结构简单，但在充分特征工程的支持下，该模型取得了令人惊喜的性能（平均 R² = 0.9974），充分验证了特征设计的有效性。其回归系数具有良好的物理可解释性，是可解释性分析页面的核心数据来源。"),

        heading3("3.1.2 随机森林"),
        para("使用 100 棵决策树的随机森林集成模型，通过自助法采样和特征随机选择增强泛化能力。该模型在留一电池交叉验证下平均 R² = 0.1539，揭示了跨电池泛化的挑战。模型训练同时输出基于 SHAP 的特征重要性排序，为可解释性分析提供支撑。"),

        heading2("3.2 深度学习模型"),

        heading3("3.2.1 Transformer 时序预测模型"),
        para("设计了专用于电池退化预测的 Transformer 架构，包含以下创新点："),
        bullet("正弦位置编码（Sinusoidal Positional Encoding）：为循环序列注入时序位置信息"),
        bullet("多头自注意力机制（4 头）：捕捉不同时间尺度的退化模式依赖关系"),
        bullet("注意力加权池化：替代传统的平均池化，让模型自主学习关键时间步的权重"),
        bullet("MC Dropout 不确定性估计：推理阶段通过 20 次前向传播获取预测置信区间"),
        para("模型结构为 2 层 Transformer 编码器，隐藏维度 64，前馈维度 128，Dropout 率 0.1。训练采用 AdamW 优化器，余弦退火学习率调度。"),

        heading3("3.2.2 物理信息神经网络（PINN）"),
        para("PINN 是本项目的核心创新之一，将电池退化的物理先验知识融入神经网络训练过程。其损失函数包含四个分量："),

        makeTable(
          ["损失分量", "物理含义", "数学形式"],
          [
            ["数据损失", "预测值与观测值的拟合误差", "MSE(y_pred, y_true)"],
            ["物理损失", "预测应接近 ECM 理论容量衰减", "MSE(y_pred, C0 + slope × cycle)"],
            ["单调性损失", "容量应随循环单调递减", "ReLU(y[t+1] - y[t]) 惩罚"],
            ["边界损失", "初始容量应接近额定值", "MSE(y[0], rated_capacity)"],
          ],
          [1800, 3200, 4360]
        ),
        emptyLine(),
        para("PINN 采用残差学习架构：网络预测的是相对于物理基线的偏差，而非绝对容量值。这大幅降低了学习难度，并保证预测结果在物理合理范围内。训练过程中各损失分量的权重通过可视化面板实时呈现。"),

        heading2("3.3 集成策略"),
        para("为充分融合 4 种基础模型的互补优势，本项目设计了 4 种集成策略："),

        heading3("3.3.1 加权集成（Weighted Ensemble）"),
        para("基于各模型在验证集上的逆 RMSE 进行权重分配。RMSE 越小的模型获得越高权重，实现简洁有效的模型融合。该策略取得了最优的平均 R² = 0.9939。"),

        heading3("3.3.2 堆叠元学习（Stacking Meta-Learner）"),
        para("将 4 个基础模型的预测值作为新特征输入 Ridge 元学习器，通过第二层模型自动学习最优组合权重。这种方法能捕捉基础模型之间的非线性互补关系。"),

        heading3("3.3.3 生命周期自适应集成"),
        para("根据电池当前 SOH 状态动态调整模型权重。早期阶段（SOH > 90%）侧重 PINN 的物理约束能力（权重 60%）；中期阶段各模型均衡贡献；晚期阶段（SOH < 80%）侧重随机森林和 Transformer 的数据驱动拟合能力。权重通过 Sigmoid 函数平滑过渡。"),

        heading3("3.3.4 物理约束集成"),
        para("在加权平均基础上引入物理一致性惩罚：对偏离 PINN 预测较远的模型施加降权处理，并通过单调性后处理确保最终预测序列符合容量递减的物理规律。"),

        heading2("3.4 模型评估"),
        para("采用留一电池交叉验证（Leave-One-Battery-Out），即每次以 7 块电池训练、1 块测试，共 8 轮，全面评估模型的跨电池泛化能力。"),

        makeTable(
          ["模型", "平均 R²", "平均 RMSE", "平均 MAE", "平均 MAPE"],
          [
            ["线性回归 (Ridge)", "0.9974", "0.0035", "0.0028", "0.19%"],
            ["随机森林", "0.1539", "0.0358", "0.0117", "0.86%"],
            ["Transformer", "-0.5103", "0.0859", "0.0546", "4.07%"],
            ["PINN", "0.4182", "0.0689", "0.0466", "3.22%"],
            ["加权集成", "0.9939", "0.0058", "0.0041", "0.27%"],
            ["堆叠元学习", "0.7195", "0.0281", "0.0150", "1.15%"],
            ["生命周期自适应", "0.7899", "0.0421", "0.0222", "1.53%"],
            ["物理约束集成", "0.6556", "0.0412", "0.0142", "0.99%"],
          ],
          [2200, 1600, 1800, 1800, 1960]
        ),
        emptyLine(),
        para("结果表明：线性回归在精心设计的特征空间中表现最优；深度学习模型（Transformer、PINN）在小样本跨电池场景下面临泛化挑战；集成策略有效融合了各模型优势，加权集成在综合指标上接近最优单模型。"),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 4: 可视化系统设计 ═══
        heading1("四、可视化系统设计"),
        para("可视化平台基于 Dash 4.x + Plotly.js 构建，采用深色奢华（Dark Luxury）设计风格，共包含 6 个功能页面。平台采用响应式布局，侧边栏导航配合 Bootstrap Icons 图标体系。"),

        heading2("4.1 车队总览仪表盘"),
        para("作为平台首页，车队总览以全局视角呈现 8 块电池的健康状态概览："),
        bullet("KPI 卡片区：车队平均 SOH（80.8%）、电池总数（8 块 / 846 循环）、最大容量衰减（41.7%）、最差电池 SOH（59.3%）"),
        bullet("车队健康矩阵：热力图展示各电池在不同循环区间的 SOH 变化，绿色到红色的渐变直观揭示退化进程"),
        bullet("最终 SOH 分布：柱状图按电池编号展示最终健康状态，80% 警告线辅助快速识别异常电池"),
        bullet("容量退化曲线：叠加展示 8 块电池的容量随循环变化趋势"),
        bullet("内阻变化趋势：电阻增长是电池老化的重要指标，与容量退化曲线形成互补视角"),
        bullet("模型性能雷达图：8 种模型的平均 R² 在极坐标系中一目了然"),

        heading2("4.2 电池详情页"),
        para("支持下拉框选择单块电池，深入分析其退化模式："),
        bullet("SOH 仪表盘：圆形仪表直观显示当前健康百分比，颜色随状态自动切换"),
        bullet("容量与 SOH 双轴图：左轴容量（Ah）、右轴 SOH（%），80% 警告阈值线"),
        bullet("特征演化 2×2 子图：内阻、放电时长、容量衰减率、电阻增长率的时序趋势"),
        bullet("放电电压曲线：范围滑块支持对比不同循环的电压-采样点曲线，颜色渐变反映时序先后"),

        heading2("4.3 数字孪生仿真器"),
        para("基于 1-RC Thevenin 等效电路模型的交互式仿真器是本平台的核心创新功能之一："),
        bullet("控制面板：滑块实时调节 C 倍率（0.5C~3C）、环境温度（10~50°C）、循环次数（0~200）"),
        bullet("ECM 参数卡片：实时显示有效 R0、R1、剩余容量、SOH、时间常数"),
        bullet("放电仿真三联图：端子电压、荷电状态（SOC）、电池温度随放电时间的协同变化"),
        bullet("老化轨迹双轴图：容量与 R0 随循环次数的演化趋势，当前循环位置标注线辅助定位"),
        para("仿真器内部通过欧拉法求解 RC 电路微分方程，集成了温度效应模型（温度系数修正电阻）和简化热模型（对流散热），实现了物理层面的闭环仿真。"),

        heading2("4.4 模型预测竞技场"),
        para("支持 8 种模型的多选对比，是评估模型性能的核心工具页面："),
        bullet("指标对比表：RMSE、MAE、R²、MAPE 四维量化评估，模型名称按专属颜色标识"),
        bullet("预测叠加图：真实值（黑色加粗）与多模型预测曲线共绘，可叠加置信区间半透明填充"),
        bullet("误差分布直方图：各模型预测误差的频率分布，半透明叠加便于对比"),
        bullet("残差散点图：残差随循环的时序分布，揭示模型在不同退化阶段的偏差模式"),
        bullet("PINN 损失分解：面积图展示数据/物理/单调性/边界四种损失在训练过程中的变化"),
        bullet("集成权重分布：分组柱状图展示 4 种集成策略对 4 个基础模型的平均权重分配"),
        bullet("Transformer 注意力热图：可视化自注意力权重，揭示模型关注的历史循环模式"),

        heading2("4.5 可解释性分析"),
        para("提供模型决策过程的透明化展示："),
        bullet("SHAP 特征重要性：水平柱状图按平均 |SHAP 值| 排序，渐变色突出主导特征"),
        bullet("线性系数分析：正负双色柱状图直观呈现各特征对容量预测的正负贡献方向"),
        bullet("跨电池热力图：归一化特征重要性矩阵，对比不同电池间的退化驱动因素差异"),
        bullet("数据效率曲线：R² 随训练数据量的变化趋势，揭示各模型的样本效率差异"),

        heading2("4.6 3D 退化景观"),
        para("将电池退化过程映射到三维空间，提供沉浸式探索体验："),
        bullet("3D 散点图：循环次数 × 内阻 × 容量构成空间坐标，按 SOH/来源/电池编号着色"),
        bullet("交互控制：着色依据切换、Z 轴变量选择、点大小调节"),
        bullet("2D 投影辅助：容量 vs. 电阻投影揭示电阻增长与容量衰减的耦合关系；SOH vs. 循环投影展示退化轨迹全貌"),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 5: 创新点分析 ═══
        heading1("五、创新点分析"),

        heading2("5.1 可视化创新（40%）"),
        bullet("深色奢华设计体系：自定义 CSS 变量体系（30+ 设计令牌），非默认模板风格"),
        bullet("6 页渐进式信息架构：从宏观车队监控到微观单体分析，从数据驱动到物理仿真"),
        bullet("3D 退化景观：创新性地将多维退化数据映射到可交互的三维空间"),
        bullet("热力图矩阵：车队健康状态的空间-时间二维编码"),
        bullet("SOH 仪表盘：将工业仪表隐喻融入健康状态展示"),

        heading2("5.2 交互创新（30%）"),
        bullet("数字孪生实时仿真：滑块驱动的 ECM 参数即时响应，所见即所得"),
        bullet("模型竞技场多选对比：自由组合 8 种模型进行实时对比分析"),
        bullet("置信区间可视化：MC Dropout 不确定性估计以半透明填充直观呈现"),
        bullet("放电曲线范围滑块：拖拽对比不同循环阶段的电压特征变化"),
        bullet("3D 视角自由旋转：鼠标拖拽探索多维数据的空间分布"),

        heading2("5.3 数据处理创新（20%）"),
        bullet("异构数据融合：统一模式设计解决 NASA (.mat) 与 CALCE (.xlsx) 的格式鸿沟"),
        bullet("20 维特征工程：涵盖电化学、温度、时序统计、衍生速率等多维度特征"),
        bullet("留一电池交叉验证：严格的跨电池泛化评估方案"),
        bullet("物理-数据融合建模：PINN 将 ECM 物理先验融入损失函数"),
        bullet("4 种集成策略：加权/堆叠/生命周期自适应/物理约束的多维度模型融合"),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 6: 系统架构 ═══
        heading1("六、系统架构与代码统计"),

        heading2("6.1 项目结构"),
        para("项目采用清晰的分层架构，按功能模块组织代码：", { indent: false }),
        bullet("src/data/：数据解析层（nasa_parser、calce_parser、unified_schema、feature_extract）"),
        bullet("src/models/：模型层（linear、rf、transformer、pinn、ensemble、ecm、trainer）"),
        bullet("src/app/：可视化层（app 主入口、chart_theme、data_loader、6 个页面模块）"),
        bullet("scripts/：训练与工具脚本"),
        bullet("data/processed/：处理后的数据文件（Parquet + JSON + 模型权重）"),

        heading2("6.2 代码规模"),
        makeTable(
          ["模块", "文件数", "总行数", "核心职责"],
          [
            ["模型层 (src/models/)", "10", "2,924", "8 种模型训练、推理、集成"],
            ["可视化层 (src/app/)", "9", "1,757", "6 页仪表盘 + 主题 + 数据加载"],
            ["数据层 (src/data/)", "4", "~600", "解析、统一、特征提取"],
            ["合计", "23+", "4,681+", "全栈可视化平台"],
          ],
          [2600, 1200, 1400, 4160]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ CHAPTER 7: 总结 ═══
        heading1("七、总结与展望"),

        heading2("7.1 项目成果"),
        para("本项目成功构建了一套完整的锂电池数字孪生可视化平台，实现了以下核心成果："),
        bullet("建立了 NASA + CALCE 双数据集的统一处理流水线，产出 846 条高质量特征记录"),
        bullet("实现了 8 种预测模型（4 基础 + 4 集成），其中加权集成模型 R² 达 0.9939"),
        bullet("构建了基于 ECM 的数字孪生仿真器，支持 C 倍率、温度、循环次数的实时交互"),
        bullet("开发了 6 页交互式仪表盘，采用深色奢华主题与 Bootstrap Icons，支持全中文界面"),
        bullet("实现了 SHAP 特征可解释性分析和 3D 退化景观可视化"),

        heading2("7.2 经验总结"),
        para("在项目实施过程中，我们获得了以下关键发现："),
        bullet("特征工程的重要性：精心设计的 20 维特征使简单的线性模型超越了复杂的深度学习模型"),
        bullet("小样本场景的挑战：仅 8 块电池的数据量限制了 Transformer 和 PINN 的泛化能力"),
        bullet("物理约束的价值：PINN 的物理损失虽未直接提升 R²，但保证了预测的物理合理性"),
        bullet("集成策略的稳健性：多模型融合有效降低了单模型的极端误差"),

        heading2("7.3 未来展望"),
        bullet("扩充数据集规模，引入更多型号和工况的电池退化数据"),
        bullet("优化 Transformer 和 PINN 在小样本场景下的迁移学习策略"),
        bullet("引入在线学习机制，支持数字孪生模型的动态参数更新"),
        bullet("开发移动端自适应布局，支持现场运维人员的便携式监控"),
        bullet("集成异常检测模块，实现电池故障的早期预警"),
      ],
    },
  ],
});

// ── Write to file ──
const OUTPUT = "/Users/zhongxudong/Desktop/大数据可视化技术/大数据可视化技术_课程报告.docx";
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("Report generated: " + OUTPUT);
});
