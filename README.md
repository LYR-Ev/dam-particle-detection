# 堰塞坝表层颗粒物质智能检测分析

> **Dam Surface Particle Intelligent Detection & Analysis**
>
> 基于 **YOLOv8 + SAM** 的无人机影像颗粒智能检测系统，自动识别堰塞坝表层石块颗粒，输出检测框、分割掩码、粒径参数（长轴/短轴/等效直径）、粒径分布直方图与统计报告。

---

## 目录

- [项目简介](#项目简介)
- [技术原理](#技术原理)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [使用方法](#使用方法)
- [输出结果说明](#输出结果说明)
- [效果展示](#效果展示)
- [评估功能（论文指标体系）](#评估功能论文指标体系)
- [注意事项](#注意事项)
- [未来改进方向](#未来改进方向)
- [许可证](#许可证)

---

## 项目简介

堰塞坝（Landslide Dam）是由滑坡、崩塌等地质灾害形成的天然坝体，其稳定性评估是防灾减灾的关键环节。坝体表层颗粒物质的粒径分布直接影响坝体的渗透性、抗冲刷能力和整体稳定性。

本项目面向**无人机（UAV）拍摄的堰塞坝 RGB 正射/倾斜影像**，利用深度学习技术实现表层石块颗粒的**全自动检测、分割与粒径测量**，为地质工程人员提供定量化的粒径分布数据，支撑堰塞坝风险评估与应急处置决策。

**核心价值：**
- 🚀 **自动化**：替代传统人工筛分或手动量测，大幅提升效率
- 🎯 **高精度**：YOLOv8 检测 + SAM 分割的组合方案，实现像素级颗粒边界提取
- 📊 **定量化**：输出每个颗粒的精确粒径参数（长轴、短轴、等效直径）及统计分布
- 📷 **非接触**：基于无人机影像，无需现场采样，适用于危险或难以到达的区域

---

## 技术原理

### 整体流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  无人机 RGB   │ -> │  YOLOv8 检测  │ -> │  SAM 分割    │ -> │  粒径测量    │
│  影像输入     │    │  颗粒定位     │    │  掩码生成     │    │  长轴/短轴   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                │
┌──────────────┐                                                 │
│  统计报告     │ <-----------------------------------------------┘
│  直方图/CSV  │
└──────────────┘
```

### 1. YOLOv8 颗粒检测（第一阶段：粗定位）

使用 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 目标检测模型对无人机影像中的石块颗粒进行快速定位。

- **模型选择**：默认使用 `yolov8x.pt`（最大版本，精度最高），也可替换为 `yolov8l.pt`、`yolov8m.pt` 等轻量版本以平衡速度与精度
- **检测输出**：每个颗粒的边界框 `(x1, y1, x2, y2)`、置信度分数、类别标签
- **可配置参数**：置信度阈值 `conf_threshold`（默认 0.25）、NMS IoU 阈值 `iou_threshold`（默认 0.45）
- **微调建议**：建议使用标注的堰塞坝颗粒数据集对 YOLOv8 进行微调，以提升特定场景下的检测精度

### 2. SAM 实例分割（第二阶段：精细分割）

使用 Meta 的 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 进行像素级颗粒分割。

- **核心思路**：将 YOLOv8 检测框作为 **Box Prompt** 输入 SAM，SAM 在框内自动生成精确的颗粒边界掩码
- **模型选择**：支持三种 ViT 骨干网络：
  | 模型 | 参数量 | 精度 | 速度 |
  |------|--------|------|------|
  | `vit_h`（默认） | ~636M | 最高 | 较慢 |
  | `vit_l` | ~308M | 较高 | 中等 |
  | `vit_b` | ~91M | 良好 | 较快 |
- **掩码后处理**：对 SAM 输出的原始掩码进行填孔、去噪、形态学开闭运算，确保掩码质量

### 3. 粒径测量（核心算法）

基于分割掩码，对每个颗粒计算以下几何参数：

#### 长轴（Long Axis）与短轴（Short Axis）

采用**主成分分析（PCA）**方法计算颗粒的最佳拟合椭圆：

1. 提取掩码的轮廓点集 `{p₁, p₂, ..., pₙ}`
2. 计算点集的协方差矩阵 `C = cov(P)`
3. 对协方差矩阵进行特征值分解，得到特征值 `λ₁, λ₂`（`λ₁ ≥ λ₂`）和对应的特征向量 `v₁, v₂`
4. 长轴 = `4 × √λ₁`，短轴 = `4 × √λ₂`（对应椭圆 2σ 边界）
5. 颗粒方向角 = `arctan2(v₁_y, v₁_x)`

#### 等效直径（Equivalent Diameter）

按面积等效原则计算：

```
D_eq = 2 × √(A / π)
```

其中 `A` 为掩码的像素面积（通过 `skimage.measure.regionprops` 计算）。

#### 其他参数

| 参数 | 含义 | 计算方式 |
|------|------|---------|
| **长宽比** | Axis Ratio | 长轴 / 短轴 |
| **面积** | Area | 掩码像素数 × 比例尺² |
| **周长** | Perimeter | 掩码轮廓长度 |
| **质心** | Centroid | 掩码区域重心坐标 |
| **方向角** | Orientation | 长轴与水平方向夹角 |

### 4. 像素到实际距离的转换

系统支持**三种比例尺标定方式**：

#### 方式一：直接指定像素分辨率（推荐）

已知无人机飞行高度 `H`（m）、相机传感器宽度 `W_sensor`（mm）、焦距 `f`（mm）、图像宽度 `W_image`（pixels）：

```
GSD (mm/pixel) = (H × W_sensor) / (f × W_image) × 1000
```

> **示例**：无人机飞行高度 100m，搭载 Phantom 4 RTK（传感器宽度 13.2mm，焦距 8.8mm），图像宽度 5472px：
> `GSD = (100 × 13.2) / (8.8 × 5472) × 1000 ≈ 0.0274 mm/pixel`

使用 `--scale` 参数传入：
```bash
python main.py --image dam.jpg --scale 0.0274
```

#### 方式二：参考线段标定

在影像中放置已知尺寸的参照物（如 1m × 1m 的标定板），或使用影像中已知尺寸的地物：

1. 测量参照物在图像中的像素长度
2. 在 `config/config.yaml` 中配置：
   ```yaml
   measurement:
     reference_length_pixels: 500    # 参照物在图像中的像素长度
     reference_length_mm: 1000       # 参照物的实际长度（毫米）
   ```

#### 方式三：使用代码接口

```python
from src.measurement.particle_measurer import ParticleMeasurer

scale = ParticleMeasurer.set_scale_from_reference(
    ref_point1=(100, 200),    # 参考点1的图像坐标
    ref_point2=(600, 200),    # 参考点2的图像坐标
    known_distance_mm=1000    # 两点之间的实际距离（毫米）
)
pipeline.run("image.jpg", scale_mm_per_pixel=scale)
```

#### 未设置比例尺时

如果未配置任何比例尺，系统将以**像素为单位**输出所有粒径参数，并在日志中给出警告提示。

---

## 项目结构

```
dam-particle-detection/
│
├── main.py                              # 程序入口脚本，命令行参数解析
├── requirements.txt                     # Python 依赖包列表
├── README.md                            # 项目说明文档（本文件）
│
├── config/
│   └── config.yaml                      # 全局配置文件（模型/测量/统计参数）
│
├── src/
│   ├── __init__.py                      # 包初始化文件
│   │
│   ├── pipeline.py                      # 主流程编排模块
│   │   └── DamParticlePipeline          # 整合 YOLO→SAM→测量→统计的完整流水线
│   │       ├── run()                    #   单张影像处理
│   │       └── run_batch()              #   批量影像处理
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py             # YOLOv8 颗粒检测模块
│   │       └── YOLODetector             # 封装 YOLOv8 模型加载与推理
│   │           ├── detect()             #   执行颗粒检测，返回边界框列表
│   │           ├── get_bbox_centers()   #   提取边界框中心坐标
│   │           └── get_bboxes_array()   #   转换为 numpy 数组
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── sam_segmentor.py             # SAM 实例分割模块
│   │       └── SAMSegmentor             # 封装 SAM 模型加载与推理
│   │           ├── set_image()          #   设置待分割图像
│   │           ├── segment_with_boxes() #   以检测框为 Prompt 生成掩码
│   │           └── refine_masks()       #   掩码后处理（去噪/填孔/平滑）
│   │
│   ├── measurement/
│   │   ├── __init__.py
│   │   └── particle_measurer.py         # 粒径测量模块
│   │       ├── ParticleMeasurement      #   数据类：存储单个颗粒的测量结果
│   │       └── ParticleMeasurer         #   粒径测量器
│   │           ├── measure()            #     批量计算所有颗粒的粒径参数
│   │           ├── _compute_geometry()  #     PCA 计算长轴/短轴
│   │           └── set_scale_from_reference()  # 参考线段比例尺标定
│   │
│   └── statistics/
│       ├── __init__.py
│       └── analyzer.py                  # 统计分析与可视化模块
│           ├── StatisticalReport        #   数据类：统计报告（均值/D10/D50/D90等）
│           └── ParticleAnalyzer         #   统计分析器
│               ├── analyze()            #     执行完整分析流程
│               ├── _plot_distribution_histograms()  # 直方图
│               ├── _plot_cumulative_distribution()  # 累计分布曲线
│               ├── _plot_summary_dashboard()        # 综合看板
│               ├── _export_csv()         #     导出 CSV 数据
│               ├── _save_report_txt()    #     保存文本报告
│               └── _save_overlay_image() #     保存检测叠加图
│
└── outputs/                             # 输出目录（自动创建），存放所有结果文件
    ├── {name}_particles.csv             #   每个颗粒的详细测量数据
    ├── {name}_statistics.csv            #   统计汇总（均值/分位数等）
    ├── {name}_report.txt                #   可读文本统计报告
    ├── {name}_histograms.png            #   三合一粒径分布直方图
    ├── {name}_cumulative.png            #   累计分布曲线
    ├── {name}_dashboard.png             #   综合信息看板
    └── {name}_overlay.png               #   检测框+分割掩码叠加可视化
```

---

## 环境配置

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 / Ubuntu 20.04 / macOS 12+ | Ubuntu 22.04 |
| **Python** | 3.9+ | 3.10 |
| **CUDA** | 11.8+（GPU 加速） | 12.1+ |
| **GPU 显存** | 8 GB（SAM vit_b） | 16 GB+（SAM vit_h） |
| **内存** | 16 GB | 32 GB+ |
| **磁盘** | 10 GB（含模型文件） | 50 GB+ |

### 安装步骤

#### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n dam_particle python=3.10 -y
conda activate dam_particle

# 或使用 venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 2. 安装 PyTorch（根据 CUDA 版本选择）

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 4. 下载预训练模型

**YOLOv8 模型**（自动下载，首次运行时）：

程序首次运行时会自动从 Ultralytics 服务器下载 YOLOv8 模型。也可手动下载：

```bash
# 下载 YOLOv8x 模型（约 130 MB）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

**SAM 模型**（需手动下载）：

从 Meta 官方仓库下载 SAM 模型权重文件，放入项目根目录：

| 模型 | 下载链接 | 大小 |
|------|---------|------|
| `vit_h`（推荐） | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | ~2.4 GB |
| `vit_l` | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | ~1.2 GB |
| `vit_b` | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | ~358 MB |

```bash
# 下载 SAM vit_h 模型
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### 5. 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "from segment_anything import sam_model_registry; print('SAM OK')"
```

---

## 使用方法

### 数据准备

1. **影像要求**：
   - 格式：JPG、PNG、TIF（推荐）
   - 色彩：RGB 三通道
   - 分辨率：建议 4000×3000 以上，确保单个颗粒占据足够像素（建议 ≥ 50px）
   - 拍摄角度：建议正射（垂直向下）拍摄，倾斜角度过大会导致比例尺不均匀

2. **比例尺准备**（任选其一）：
   - 计算 GSD（地面采样距离），参见[像素到实际距离的转换](#4-像素到实际距离的转换)
   - 在影像中放置已知尺寸的参照物

3. **（可选）标注数据用于模型微调**：
   - 使用 LabelImg、Labelme 等工具标注颗粒边界框
   - 格式：YOLO 格式（class_id, cx, cy, w, h）或 COCO 格式

### 单张影像处理

```bash
# 基本用法（不指定比例尺，输出像素单位）
python main.py --image path/to/uav_image.jpg

# 指定像素分辨率（mm/pixel）
python main.py --image path/to/uav_image.jpg --scale 0.35

# 指定输出目录
python main.py --image path/to/uav_image.jpg --scale 0.35 --output results/

# 使用 CPU
python main.py --image path/to/uav_image.jpg --device cpu

# 使用自定义配置文件
python main.py --config config/custom.yaml --image path/to/uav_image.jpg
```

### 批量处理

```bash
# 处理整个目录下的所有影像
python main.py --dir ./uav_images/ --scale 0.35

# 指定输出目录
python main.py --dir ./uav_images/ --scale 0.35 --output batch_results/
```

### Python API 调用

```python
from src.pipeline import DamParticlePipeline

pipeline = DamParticlePipeline("config/config.yaml")

# 单张处理
result = pipeline.run("uav_image.jpg", scale_mm_per_pixel=0.35)
print(f"检测到 {len(result['measurements'])} 个颗粒")

# 批量处理
results = pipeline.run_batch("./uav_images/", scale_mm_per_pixel=0.35)
```

### 模型微调（可选，提升精度）

```bash
# 使用标注数据微调 YOLOv8
yolo train data=dam_particles.yaml model=yolov8x.pt epochs=100 imgsz=1280
```

---

## 输出结果说明

每张影像处理完成后，在 `outputs/` 目录下生成以下文件：

### 1. 颗粒测量数据 — `{name}_particles.csv`

包含每个颗粒的详细测量结果，字段说明：

| 字段名 | 含义 | 单位 |
|--------|------|------|
| `particle_id` | 颗粒编号 | — |
| `x1, y1, x2, y2` | 边界框坐标 | 像素 |
| `area_mm2` | 颗粒实际面积 | mm² |
| `long_axis_mm` | 长轴长度 | mm |
| `short_axis_mm` | 短轴长度 | mm |
| `equivalent_diameter_mm` | 等效直径 | mm |
| `axis_ratio` | 长宽比（长轴/短轴） | — |
| `centroid_x, centroid_y` | 颗粒质心坐标 | 像素 |
| `orientation_deg` | 颗粒方向角 | 度 (°) |

### 2. 统计汇总 — `{name}_statistics.csv`

各粒径指标的统计量（均值、标准差、D10/D25/D50/D75/D90 分位数等）。

### 3. 文本报告 — `{name}_report.txt`

人类可读的统计报告，包含：
- 检测颗粒总数
- 长轴、短轴、等效直径、面积的均值/标准差/最小值/最大值/D10/D25/D50/D75/D90

### 4. 可视化图表

#### 粒径分布直方图 — `{name}_histograms.png`

三合一分布图，分别展示长轴、短轴、等效直径的频率分布，标注均值和中位数。

#### 累计分布曲线 — `{name}_cumulative.png`

长轴、短轴、等效直径的累计频率曲线，标注 D10/D50/D90 参考线。

#### 综合信息看板 — `{name}_dashboard.png`

四合一信息面板：
- **左上**：长轴 vs 短轴散点图（颜色编码等效直径）
- **右上**：等效直径箱线图
- **左下**：长宽比分布直方图
- **右下**：粒径分级统计柱状图（<20 / 20-50 / 50-100 / 100-200 / 200-500 / >500 mm）

#### 检测叠加图 — `{name}_overlay.png`

原始影像上叠加半透明分割掩码和检测框，每个颗粒标注编号。

---

## 效果展示

> **请在下方添加您的测试结果截图。**
>
> 建议展示：
> 1. 原始无人机影像
> 2. 检测叠加图（`overlay.png`）
> 3. 粒径分布直方图（`histograms.png`）
> 4. 综合信息看板（`dashboard.png`）

| 原始影像 | 检测叠加图 |
|:--------:|:----------:|
| （待添加） | （待添加） |

| 粒径分布直方图 | 综合信息看板 |
|:------------:|:----------:|
| （待添加） | （待添加） |

---

## 评估功能（论文指标体系）

本系统集成了2024年《Minerals》顶刊论文《Identification of Rock Fragments after Blasting by Using Deep Learning-Based Segment Anything Model》的完整评估指标体系。

### 评估指标概览

| 指标类别 | 指标名称 | 论文公式 | 对应模块 |
|---------|---------|---------|---------|
| **像素级** | PA（像素精度） | 公式(1) | `metrics.py:pixel_accuracy()` |
| **像素级** | mIOU（平均交并比） | 公式(2) | `metrics.py:mean_iou()` |
| **像素级** | Dice 系数 | 公式(3) | `metrics.py:dice_coefficient()` |
| **粒径级** | 等效圆直径 | 公式(4) | `metrics.py:equivalent_circle_diameter()` |
| **粒径级** | Rosin-Rammler 分布拟合 | 公式(5)-(9) | `metrics.py:fit_rosin_rammler()` |
| **回归** | MAE / RMSE / R² / MRE | — | `metrics.py:compute_mae/rmse/r_squared/mre()` |

### 评估模式使用方法

```bash
# 评估模式：将 SAM 预测结果与人工标注真值对比
python main.py --evaluate \
    --image dam_uav_001.jpg \
    --scale 0.35 \
    --ground-truth ./ground_truth/ \
    --output-report ./reports/
```

### 标注数据准备

`--ground-truth` 目录支持两种格式：

**格式一：实例掩码图（推荐）**
```
ground_truth/
└── dam_uav_001_mask.png    # 每个像素值为颗粒实例 ID（=0 为背景）
```

**格式二：单独掩码文件**
```
ground_truth/
├── dam_uav_001_mask_0.png
├── dam_uav_001_mask_1.png
└── dam_uav_001_mask_2.png
```

**（可选）粒径真值 CSV**
如果已有粒径测量结果，可直接提供 CSV：
```
ground_truth/
└── dam_uav_001_particles.csv   # 格式与系统输出的 particles.csv 一致
```

### 评估报告输出

| 输出文件 | 说明 | 论文对应 |
|---------|------|---------|
| `{name}_evaluation_report.txt` | 文本格式评估报告 | Table 1 + Table 4 格式 |
| `{name}_evaluation_pixel_metrics.csv` | 像素级指标 (PA/mIOU/Dice) | Table 1 / Table 3 |
| `{name}_evaluation_char_sizes.csv` | 特征粒径对比 (X10~X100) | Table 4 |
| `{name}_evaluation_grain_level.csv` | 粒径级指标 + RR 拟合参数 | Figure 15 |
| `{name}_size_distribution.png` | 粒径分布直方图 + RR PDF 拟合 | Figure 6/10 |
| `{name}_cumulative_rr.png` | 累计通过率曲线 + RR CDF + X10~X100 | Figure 7/11/14 |
| `{name}_char_sizes_comparison.png` | 特征粒径对比柱状图 | Table 4 可视化 |
| `{name}_metrics_bars.png` | 评估指标总览柱状图 | — |

### 验证评估功能

```bash
python test_evaluation.py
```

### 核心论文公式实现

**公式(1) — 像素精度 (PA)**:
```
PA = Σ p_ii / Σ Σ p_ij
```

**公式(2) — 平均交并比 (mIOU)**:
```
mIOU = (1/(n+1)) × Σ p_ii / (Σ_j p_ij + Σ_j p_ji - p_ii)
```

**公式(3) — Dice 系数**:
```
Dice = 2|X ∩ Y| / (|X| + |Y|)
```

**公式(4) — 等效圆直径**:
```
D = 2√(S/π)
```

**公式(8) — Rosin-Rammler 累积分布**:
```
R(x) = 1 - exp(-0.693 × (X/Xm)^n)
```

**公式(9) — 均匀性指数**:
```
n = 0.842 / (ln(k80) - ln(k50))
```

---

## 注意事项

### 硬件要求

- **GPU 显存**：SAM vit_h 模型需要约 8 GB 显存（单张 1024×1024 影像）。处理更大影像时会自动切片，但显存需求可能增加。若显存不足，可切换为 `vit_b` 模型。
- **CPU 模式**：SAM 在 CPU 上运行非常缓慢，单张影像可能耗时数分钟，仅建议在无 GPU 环境下进行小规模测试。
- **内存占用**：处理高分辨率影像（如 8000×6000）时，内存占用可能超过 16 GB。

### 数据标注建议

如使用 YOLOv8 预训练权重不满足精度要求，建议进行微调：
- 标注 200-500 张具有代表性的堰塞坝影像
- 标注类别统一为 `particle`（类别 ID: 0），或根据实际需求细分 `boulder`（漂石）/`cobble`（卵石）/`gravel`（砾石）
- 使用 YOLO 格式：`class_id cx cy w h`（归一化到 0-1）
- 建议使用 LabelImg 或 [Roboflow](https://roboflow.com) 进行标注

### 精度影响因素

| 因素 | 影响 | 建议 |
|------|------|------|
| 影像分辨率 | 低分辨率导致小颗粒漏检 | GSD < 5 mm/pixel，单个颗粒 ≥ 50 px |
| 光照条件 | 阴影/过曝降低检测精度 | 阴天或均匀光照条件下拍摄 |
| 颗粒重叠 | 密集堆叠导致分割粘连 | 调整 YOLO NMS 阈值，增加训练数据 |
| 植被遮挡 | 部分颗粒不可见 | 预处理时尽量选择裸露地表区域 |
| 比例尺精度 | 直接影响粒径测量绝对值 | 使用 RTK 无人机或地面控制点精确定标 |

### 常见问题排查

<details>
<summary><b>Q: 提示 "CUDA out of memory"</b></summary>

**解决方法**：
1. 在 `config/config.yaml` 中将 SAM 模型切换为 `vit_b` 或 `vit_l`
2. 降低输入影像分辨率（如缩小到 2048px 宽）
3. 使用 `--device cpu` 回退到 CPU（速度较慢）
</details>

<details>
<summary><b>Q: 检测到的颗粒数量太少</b></summary>

**解决方法**：
1. 降低 `conf_threshold`（如 0.1）
2. 微调 YOLOv8 模型
3. 检查影像中小颗粒是否占据足够像素
</details>

<details>
<summary><b>Q: 分割掩码边界不准确</b></summary>

**解决方法**：
1. 使用 SAM vit_h 模型（精度最高）
2. 检查 SAM 模型权重文件是否完整
3. 调整 `mask_threshold` 等 SAM 参数
</details>

<details>
<summary><b>Q: 中文图表显示为方框</b></summary>

**解决方法**：
```bash
# Ubuntu
sudo apt install fonts-wqy-microhei

# 或在代码中指定可用字体
# 修改 src/statistics/analyzer.py 中的 plt.rcParams["font.sans-serif"]
```
</details>

---

## 未来改进方向

- [ ] **SAM 2**：升级到 Meta 最新发布的 SAM 2，支持视频流处理和更高效的推理
- [ ] **YOLOv8 微调模块**：内置训练脚本，支持一键微调
- [ ] **多尺度推理**：使用 SAHI 框架对大尺寸影像进行切片推理，提升小目标检测精度
- [ ] **3D 粒径重建**：结合多视角无人机影像和 SfM（Structure from Motion）进行三维颗粒重建
- [ ] **Web 界面**：开发基于 Gradio 或 Streamlit 的 Web 前端，降低使用门槛
- [ ] **实时处理**：优化推理速度，支持无人机实时图传的在线检测
- [ ] **颗粒级配曲线**：自动生成级配曲线（Grain Size Distribution Curve），与筛分试验结果对比
- [ ] **多类别识别**：支持区分不同岩性/风化程度的颗粒
- [ ] **时空变化分析**：支持多期影像对比，分析颗粒分布的时空变化

---

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。

使用的第三方模型遵循各自的许可证：
- [YOLOv8](https://github.com/ultralytics/ultralytics)：AGPL-3.0
- [SAM](https://github.com/facebookresearch/segment-anything)：Apache-2.0

---

> **引用**：如果您在研究中使用了本项目，请注明来源。
>
> **联系方式**：如有问题或建议，欢迎提交 GitHub Issue。