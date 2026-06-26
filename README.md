# 堰塞坝表层颗粒物质智能检测分析

> **Dam Surface Particle Intelligent Detection & Analysis**
>
> 基于 **YOLOv8 + SAM** 的无人机影像颗粒智能检测系统，自动识别堰塞坝表层石块颗粒，输出检测框、分割掩码、粒径参数（长轴/短轴/等效直径）、粒径分布直方图与统计报告。

---

## 目录

- [项目简介](#项目简介)
- [项目情况](#项目情况)
- [技术原理](#技术原理)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [依赖安装](#依赖安装)
- [模型训练](#模型训练)
- [正式运行识别](#正式运行识别)
- [模型评估](#模型评估)
- [输出结果说明](#输出结果说明)
- [注意事项](#注意事项)
- [常见问题排查](#常见问题排查)
- [未来改进方向](#未来改进方向)
- [许可证](#许可证)

---

## 项目简介

堰塞坝（Landslide Dam）是由滑坡、崩塌等地质灾害形成的天然坝体，其稳定性评估是防灾减灾的关键环节。坝体表层颗粒物质的粒径分布直接影响坝体的渗透性、抗冲刷能力和整体稳定性。

本项目面向**无人机（UAV）拍摄的堰塞坝 RGB 正射/倾斜影像**，利用深度学习技术实现表层石块颗粒的**全自动检测、分割与粒径测量**，为地质工程人员提供定量化的粒径分布数据，支撑堰塞坝风险评估与应急处置决策。

**核心价值：**
- **自动化**：替代传统人工筛分或手动量测，大幅提升效率
- **高精度**：YOLOv8 检测 + SAM 分割的组合方案，实现像素级颗粒边界提取
- **定量化**：输出每个颗粒的精确粒径参数（长轴、短轴、等效直径）及统计分布
- **非接触**：基于无人机影像，无需现场采样，适用于危险或难以到达的区域

---

## 项目情况

### 已完成功能

| 模块 | 状态 | 说明 |
|------|------|------|
| YOLOv8 颗粒检测 | ✅ 已完成 | 支持预训练权重和自训练权重 |
| SAM 实例分割 | ✅ 已完成 | 支持 vit_h / vit_l / vit_b 三种骨干网络 |
| PCA 粒径测量 | ✅ 已完成 | 长轴/短轴/等效直径/面积/周长/方向角 |
| 统计分析 | ✅ 已完成 | 直方图/累计曲线/看板/CSV/TXT 报告 |
| 模型训练脚本 | ✅ 已完成 | 两阶段训练 + OOM 自动降级 + 权重自动部署 |
| 评估指标体系 | ✅ 已完成 | 集成 2024《Minerals》论文公式 1-9 |
| OpenCV Fallback | ✅ 已完成 | 无 GPU 环境下基于分水岭算法的备选方案 |

### 训练成果

本项目已在合成数据集上完成 YOLOv8 训练：

| 模型 | 轮数 | mAP50 | mAP50-95 | 状态 |
|------|------|-------|----------|------|
| yolov8n | 50 | 0.537 | 0.217 | ✅ 已部署到 `models/best.pt` |
| yolov8x | 38 (早停) | 0.373 | 0.155 | 数据集过小导致模型崩溃 |

### 引用论文

本项目评估指标体系引用自 2024 年《Minerals》期刊论文：
> *Identification of Rock Fragments after Blasting by Using Deep Learning-Based Segment Anything Model*

论文中的 PA、mIOU、Dice、等效圆直径、Rosin-Rammler 分布拟合等公式已完整实现于 `metrics.py`。

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

- **模型选择**：默认使用训练后的 `models/best.pt`，也可替换为 `yolov8x.pt`、`yolov8n.pt` 等预训练权重
- **检测输出**：每个颗粒的边界框 `(x1, y1, x2, y2)`、置信度分数、类别标签
- **可配置参数**：置信度阈值 `conf_threshold`（默认 0.25）、NMS IoU 阈值 `iou_threshold`（默认 0.45）

### 2. SAM 实例分割（第二阶段：精细分割）

使用 Meta 的 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 进行像素级颗粒分割。

- **核心思路**：将 YOLOv8 检测框作为 **Box Prompt** 输入 SAM，SAM 在框内自动生成精确的颗粒边界掩码
- **模型选择**：

  | 模型 | 参数量 | 精度 | 速度 |
  |------|--------|------|------|
  | `vit_h`（默认） | ~636M | 最高 | 较慢 |
  | `vit_l` | ~308M | 较高 | 中等 |
  | `vit_b` | ~91M | 良好 | 较快 |

### 3. 粒径测量（PCA 算法）

基于分割掩码，采用**主成分分析（PCA）**计算颗粒的最佳拟合椭圆：

1. 提取掩码轮廓点集，计算协方差矩阵
2. 特征值分解得到 `λ₁, λ₂`（`λ₁ ≥ λ₂`）
3. 长轴 = `4 × √λ₁`，短轴 = `4 × √λ₂`
4. 等效直径：`D_eq = 2 × √(A / π)`（A 为掩码面积）

### 4. 像素到实际距离的转换

支持三种比例尺标定方式：

**方式一：直接指定 GSD（推荐）**
```bash
python main.py --image dam.jpg --scale 0.0274  # mm/pixel
```

**方式二：参考线段标定**（在 `config/config.yaml` 中配置）
```yaml
measurement:
  reference_length_pixels: 500    # 参照物像素长度
  reference_length_mm: 1000        # 实际长度（毫米）
```

**方式三：未设置比例尺** — 系统以像素为单位输出所有粒径参数，并给出警告。

---

## 项目结构

```
dam-particle-detection/
│
├── main.py                              # 程序入口脚本（检测/评估两种模式）
├── train_yolo.py                        # YOLOv8 训练脚本（两阶段 + OOM 自动降级）
├── run_fallback.py                      # OpenCV 备选检测方案（无 GPU 时使用）
├── metrics.py                           # 论文评估指标实现（公式 1-9）
├── test_evaluation.py                   # 评估功能单元测试
├── setup_env.ps1                        # 一键环境配置脚本（Windows PowerShell）
├── requirements.txt                     # Python 依赖列表
├── README.md                            # 项目说明文档
│
├── config/
│   └── config.yaml                      # 全局配置（模型/测量/统计参数）
│
├── models/
│   └── best.pt                          # 训练生成的最佳权重（自动部署）
│
├── dataset/                             # 训练数据集（YOLO 格式）
│   ├── data.yaml
│   ├── images/
│   └── labels/
│
├── runs/                                # 训练运行输出（自动生成，已 gitignore）
│
├── src/
│   ├── pipeline.py                      # 主流程编排（YOLO→SAM→测量→统计）
│   ├── detection/yolo_detector.py       # YOLOv8 检测模块
│   ├── segmentation/sam_segmentor.py    # SAM 分割模块
│   ├── measurement/particle_measurer.py # 粒径测量模块（PCA）
│   └── statistics/analyzer.py            # 统计分析与可视化
│
└── outputs/                             # 输出目录（自动创建，已 gitignore）
```

---

## 环境准备

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 / Ubuntu 20.04 / macOS 12+ | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.9+ | **3.12**（CUDA 训练必须） |
| **CUDA** | 11.8+（GPU 加速） | 12.1+ |
| **GPU 显存** | 4 GB（训练 yolov8n） | 8 GB+（SAM vit_h + yolov8x） |
| **内存** | 8 GB | 16 GB+ |
| **磁盘** | 5 GB（代码+依赖） | 10 GB+（含模型权重） |

> **重要**：Python 3.14 **没有** CUDA 版 PyTorch wheel，训练必须使用 Python 3.12。检测推理可使用 Python 3.9+。

### 方式一：一键环境配置（推荐，Windows）

项目提供 `setup_env.ps1` 脚本，自动完成全部环境搭建：

```powershell
cd d:\19701\Source\Repos\dam-particle-detection
.\setup_env.ps1
```

脚本自动完成：
1. 检测/安装 Python 3.12（通过 winget）
2. 创建虚拟环境 `.venv312`
3. 安装 CUDA 12.4 版 PyTorch（`torch torchvision`，约 2.5 GB）
4. 安装 ultralytics 及项目全部依赖
5. 验证 CUDA 可用性

> **注意**：`setup_env.ps1` 为纯 ASCII 英文脚本，以避免 Windows PowerShell 5.1 的 GBK 编码解析问题。

### 方式二：手动环境配置

#### 1. 安装 Python 3.12

```powershell
# Windows (winget)
winget install Python.Python.3.12 --scope user

# 或从官网下载: https://www.python.org/downloads/release/python-3120/
```

#### 2. 创建虚拟环境

```bash
# 在项目根目录下
python -m venv .venv312

# Windows 激活
.venv312\Scripts\activate

# Linux/Mac 激活
source .venv312/bin/activate
```

#### 3. 下载预训练模型

**YOLOv8 模型**（首次运行自动下载，也可手动下载）：
```bash
# 下载 yolov8x 模型（约 130 MB）
# 放到项目根目录
```

**SAM 模型**（需手动下载，放到项目根目录）：

| 模型 | 下载链接 | 大小 |
|------|---------|------|
| `vit_h`（推荐） | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | ~2.4 GB |
| `vit_l` | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | ~1.2 GB |
| `vit_b` | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | ~358 MB |

---

## 依赖安装

### 方法一：pip 安装（推荐）

```bash
# 确保已激活虚拟环境
.venv312\Scripts\activate

# 1. 安装 CUDA 版 PyTorch（CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. 安装项目依赖
pip install -r requirements.txt
```

> 如果使用其他 CUDA 版本：
> - CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
> - CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
> - 仅 CPU: `--index-url https://download.pytorch.org/whl/cpu`

### 方法二：setup_env.ps1 自动安装

运行 `setup_env.ps1` 会自动安装所有依赖，无需手动操作。

### 依赖列表

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| `torch` | 2.0.0 | 深度学习框架（YOLOv8 + SAM） |
| `torchvision` | 0.15.0 | 图像处理工具 |
| `ultralytics` | 8.0.0 | YOLOv8 检测框架 |
| `segment-anything` | 1.0 | SAM 分割模型 |
| `opencv-python` | 4.8.0 | 图像处理 + Fallback 检测 |
| `numpy` | 1.24.0 | 数值计算 |
| `scipy` | 1.10.0 | 科学计算 |
| `scikit-image` | 0.21.0 | 图像测量 |
| `pandas` | 2.0.0 | 数据处理 |
| `matplotlib` | 3.7.0 | 可视化绘图 |
| `pyyaml` | 6.0 | 配置文件解析 |
| `tqdm` | — | 进度条（训练时安装） |

### 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "from segment_anything import sam_model_registry; print('SAM OK')"
```

---

## 模型训练

本项目提供完整的 YOLOv8 训练流程，支持两阶段训练策略：先用 yolov8n 快速验证流程，再用 yolov8x 进行高精度训练。

### 训练流程概览

```
环境配置 -> 数据集准备 -> 快速验证训练(yolov8n) -> 高精度训练(yolov8x) -> 部署权重
```

### 步骤 1：准备数据集

数据集采用 YOLO 格式，目录结构：

```
dataset/
├── data.yaml          # 数据集配置
├── images/            # 图像文件（.jpg/.png）
│   ├── 000000.png
│   ├── 000001.png
│   └── 000002.png
└── labels/            # 标签文件（每行: class cx cy w h，归一化坐标）
    ├── 000000.txt
    ├── 000001.txt
    └── 000002.txt
```

`data.yaml` 配置：

```yaml
path: d:/19701/Source/Repos/dam-particle-detection/dataset
train: images
val: images
test: images

nc: 1              # 类别数
names: ['rock']    # 类别名
```

### 步骤 2：快速验证训练（yolov8n）

使用轻量模型快速验证训练流程：

```bash
.\.venv312\Scripts\python.exe train_yolo.py --stage quick
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolov8n.pt | 最轻量（3M 参数），快速验证 |
| 轮数 | 50 | 基础训练 |
| 图像尺寸 | 1024 px | 输入分辨率 |
| batch | 8 | 批大小 |
| patience | 20 | 早停耐心值 |

### 步骤 3：高精度训练（yolov8x）

```bash
.\.venv312\Scripts\python.exe train_yolo.py --stage final
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolov8x.pt | 最大版本（68M 参数），精度最高 |
| 轮数 | 150 | 充分训练 |
| 图像尺寸 | 1024 px | 输入分辨率 |
| batch | 4 | 批大小（大模型显存占用高） |
| patience | 30 | 早停耐心值 |

### 步骤 4：自定义训练参数

覆盖默认参数进行自定义训练：

```bash
# 使用 yolov8s 模型，100 轮，batch 4
.\.venv312\Scripts\python.exe train_yolo.py --model yolov8s.pt --epochs 100 --batch 4

# 指定自定义数据集路径和图像尺寸
.\.venv312\Scripts\python.exe train_yolo.py --stage final --data D:/path/to/data.yaml --imgsz 1280

# 使用 yolov8n 快速验证，指定数据集
.\.venv312\Scripts\python.exe train_yolo.py --stage quick --data D:/path/to/data.yaml
```

完整命令行参数：

```
--stage      quick / final        训练阶段（默认 final）
--model      yolov8n.pt 等        自定义预训练模型（覆盖 stage 默认值）
--epochs     50 / 150 / 自定义     训练轮数
--batch      8 / 4 / 自定义        批大小
--imgsz      1024 / 自定义         图像尺寸
--data       data.yaml 路径        数据集配置文件
```

### OOM 自动降级机制

训练脚本针对小显存 GPU（如 RTX 3050 4GB）内置显存溢出自动重试：

```
尝试 1/4: batch=原值,  imgsz=原值       ← 默认参数
尝试 2/4: batch=原值/2, imgsz=原值       ← batch 减半
尝试 3/4: batch=原值/2, imgsz=原值/2    ← imgsz 减半
尝试 4/4: batch=1,     imgsz=原值/2     ← 最小配置
```

遇到 `CUDA out of memory` 错误时自动清理显存并降级重试，无需手动调整参数。

### 步骤 5：自动部署训练权重

训练完成后，脚本自动执行：

1. 将 `runs/train/<name>/weights/best.pt` 复制到 `models/best.pt`
2. 更新 `config/config.yaml` 中的 `yolo.model_path` 为 `./models/best.pt`
3. 输出训练摘要（最终 mAP50、mAP50-95、训练轮数）

### 训练结果

| 阶段 | 模型 | 实际轮数 | mAP50 | mAP50-95 | 说明 |
|------|------|---------|-------|----------|------|
| Quick | yolov8n | 50 | 0.537 | 0.217 | 训练稳定，**已部署** |
| Final | yolov8x | 38 (早停) | 0.373 | 0.155 | 数据集过小导致模型崩溃 |

> **结论**：当数据集较小时（< 50 张），推荐使用 yolov8n。大模型容易在小数据集上过拟合后崩溃。扩充数据集是提升精度的最有效途径。

### 训练输出文件

`runs/train/<name>/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | 最佳权重（按验证 mAP 选择） |
| `weights/last.pt` | 最后一轮权重 |
| `results.csv` | 每 epoch 的训练/验证指标 |
| `results.png` | 训练曲线图 |
| `confusion_matrix.png` | 混淆矩阵 |
| `BoxPR_curve.png` | PR 曲线 |
| `BoxF1_curve.png` | F1 曲线 |
| `train_batch*.jpg` | 训练批次可视化 |
| `val_batch*_pred.jpg` | 验证预测可视化 |

---

## 正式运行识别

### 单张影像识别

```bash
# 基本用法（不指定比例尺，输出像素单位）
python main.py --image path/to/uav_image.jpg

# 指定像素分辨率（mm/pixel）
python main.py --image path/to/uav_image.jpg --scale 0.35

# 指定输出目录
python main.py --image path/to/uav_image.jpg --scale 0.35 --output results/

# 使用 CPU（无 GPU 时）
python main.py --image path/to/uav_image.jpg --device cpu

# 使用训练后的权重（config.yaml 已自动配置）
.\.venv312\Scripts\python.exe main.py --image path/to/uav_image.jpg --scale 0.5
```

### 批量影像识别

```bash
# 处理整个目录下的所有影像
python main.py --dir ./uav_images/ --scale 0.35

# 指定输出目录
python main.py --dir ./uav_images/ --scale 0.35 --output batch_results/
```

### 自定义路径示例

```bash
# 识别 D 盘指定路径下的单张影像
python main.py --image "D:/2026/模式识别课程设计/资源/数据集/dataset-build-self/images/10.jpg" --scale 0.5

# 识别 D 盘指定目录下的所有影像
python main.py --dir "D:/2026/模式识别课程设计/资源/数据集/dataset-build-self/images/" --scale 0.5

# 使用虚拟环境中的 Python 执行
.\.venv312\Scripts\python.exe main.py --image "D:/path/to/image.jpg" --scale 0.5 --output "D:/path/to/output/"
```

### 使用自定义配置文件

```bash
python main.py --config config/custom.yaml --image path/to/image.jpg
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

### Fallback 模式（无 GPU 时）

当 YOLOv8 + SAM 不可用时（无 GPU 或未下载模型权重），可使用基于 OpenCV 分水岭算法的备选方案：

```bash
# 单张影像
python run_fallback.py --input path/to/image.jpg --scale 0.5

# 目录批量处理
python run_fallback.py --input path/to/images/ --scale 0.5
```

### 命令行参数总览

```
main.py 参数:
  --image, -i PATH         输入影像路径（与 --dir 互斥）
  --dir, -d PATH           批量处理的影像目录路径（与 --image 互斥）
  --config, -c PATH        配置文件路径（默认: config/config.yaml）
  --scale, -s FLOAT        像素分辨率 mm/pixel（覆盖配置文件）
  --output, -o PATH        输出目录（默认: outputs/）
  --device {cuda,cpu}      计算设备（覆盖配置文件）
  --evaluate               进入评估模式
  --ground-truth PATH      标注数据文件夹路径（评估模式必需）
  --output-report PATH     评估报告输出路径

train_yolo.py 参数:
  --stage {quick,final}    训练阶段（默认 final）
  --model PATH             自定义预训练模型
  --epochs N               训练轮数
  --batch N                批大小
  --imgsz N                图像尺寸
  --data PATH              data.yaml 路径
```

---

## 模型评估

本系统集成 2024 年《Minerals》期刊论文的完整评估指标体系，支持将预测结果与人工标注真值对比，生成论文格式的评估报告。

### 评估指标

| 指标类别 | 指标名称 | 论文公式 | 说明 |
|---------|---------|---------|------|
| **像素级** | PA（像素精度） | 公式(1) | 预测正确的像素占比 |
| **像素级** | mIOU（平均交并比） | 公式(2) | 预测与真值的重叠度 |
| **像素级** | Dice 系数 | 公式(3) | 相似度度量 |
| **粒径级** | 等效圆直径 | 公式(4) | `D = 2√(S/π)` |
| **粒径级** | Rosin-Rammler 分布 | 公式(5)-(9) | 粒径分布拟合 |
| **回归** | MAE / RMSE | — | 平均绝对误差 / 均方根误差 |
| **回归** | R² / MRE | — | 决定系数 / 平均相对误差 |

### 运行评估

```bash
# 评估模式：将 SAM 预测结果与标注真值对比
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
└── dam_uav_001_mask.png    # 每个像素值为颗粒实例 ID（0=背景）
```

**格式二：单独掩码文件**
```
ground_truth/
├── dam_uav_001_mask_0.png
├── dam_uav_001_mask_1.png
└── dam_uav_001_mask_2.png
```

**（可选）粒径真值 CSV**
```
ground_truth/
└── dam_uav_001_particles.csv   # 格式与系统输出的 particles.csv 一致
```

### 评估报告输出

| 输出文件 | 说明 | 论文对应 |
|---------|------|---------|
| `{name}_evaluation_report.txt` | 文本评估报告 | Table 1 + Table 4 |
| `{name}_evaluation_pixel_metrics.csv` | 像素级指标 | Table 1 / Table 3 |
| `{name}_evaluation_char_sizes.csv` | 特征粒径对比 | Table 4 |
| `{name}_evaluation_grain_level.csv` | 粒径级指标 + RR 拟合 | Figure 15 |
| `{name}_size_distribution.png` | 粒径分布 + RR PDF 拟合 | Figure 6/10 |
| `{name}_cumulative_rr.png` | 累计通过率 + RR CDF | Figure 7/11/14 |
| `{name}_char_sizes_comparison.png` | 特征粒径对比柱状图 | Table 4 |
| `{name}_metrics_bars.png` | 评估指标总览柱状图 | — |

### 运行单元测试

```bash
# 验证评估功能（无需 GPU / 模型 / 标注数据，使用模拟数据）
python test_evaluation.py
```

### 核心论文公式

**公式(1) — 像素精度**: `PA = Σ p_ii / Σ Σ p_ij`

**公式(2) — 平均交并比**: `mIOU = (1/(n+1)) × Σ p_ii / (Σ_j p_ij + Σ_j p_ji - p_ii)`

**公式(3) — Dice 系数**: `Dice = 2|X ∩ Y| / (|X| + |Y|)`

**公式(4) — 等效圆直径**: `D = 2√(S/π)`

**公式(8) — Rosin-Rammler 累积分布**: `R(x) = 1 - exp(-0.693 × (X/Xm)^n)`

---

## 输出结果说明

每张影像处理完成后，在 `outputs/` 目录下生成：

### 颗粒测量数据 — `{name}_particles.csv`

| 字段名 | 含义 | 单位 |
|--------|------|------|
| `particle_id` | 颗粒编号 | — |
| `x1, y1, x2, y2` | 边界框坐标 | 像素 |
| `area_mm2` | 颗粒面积 | mm² |
| `long_axis_mm` | 长轴长度 | mm |
| `short_axis_mm` | 短轴长度 | mm |
| `equivalent_diameter_mm` | 等效直径 | mm |
| `axis_ratio` | 长宽比 | — |
| `centroid_x, centroid_y` | 质心坐标 | 像素 |
| `orientation_deg` | 方向角 | 度 (°) |

### 可视化图表

| 文件 | 说明 |
|------|------|
| `{name}_histograms.png` | 三合一粒径分布直方图（长轴/短轴/等效直径） |
| `{name}_cumulative.png` | 累计分布曲线（标注 D10/D50/D90） |
| `{name}_dashboard.png` | 综合信息看板（散点/箱线/分级统计） |
| `{name}_overlay.png` | 检测框 + 分割掩码叠加图 |
| `{name}_report.txt` | 文本统计报告（均值/标准差/分位数） |
| `{name}_statistics.csv` | 统计汇总 CSV |

---

## 注意事项

### Python 版本

- **训练必须使用 Python 3.12**：Python 3.14 没有 CUDA 版 PyTorch wheel（cu121/cu124 索引均不提供 cp314 包）
- **检测推理可用 Python 3.9+**：CPU 模式或已有 torch 环境即可
- 推荐使用 `setup_env.ps1` 自动安装 Python 3.12

### GPU 显存

- **SAM vit_h** 需要约 8 GB 显存，显存不足时切换 `vit_b`
- **YOLOv8x 训练** batch=4 + imgsz=1024 在 4GB GPU 上可能 OOM，脚本会自动降级
- **推理时** 如显存不足，使用 `--device cpu` 回退（速度较慢）

### 数据集要求

- **训练数据**：建议至少 200-500 张标注影像才能有效微调
- **小数据集（< 50 张）**：使用 yolov8n，避免 yolov8x 过拟合崩溃
- **标注格式**：YOLO 格式 `class_id cx cy w h`（归一化到 0-1）
- **图像分辨率**：建议 GSD < 5 mm/pixel，单个颗粒 ≥ 50 px

### 比例尺精度

- 比例尺直接影响粒径测量的绝对值
- 推荐使用 RTK 无人机或地面控制点精确定标
- 未设置比例尺时系统以像素为单位输出

### 模型权重文件

- 模型权重文件（`.pt` / `.pth`）体积较大，已加入 `.gitignore`
- SAM vit_h 模型约 2.4 GB，需手动下载
- YOLOv8 模型首次运行时自动下载

---

## 常见问题排查

<details>
<summary><b>Q: 提示 "CUDA out of memory"</b></summary>

1. 在 `config/config.yaml` 中将 SAM 模型切换为 `vit_b` 或 `vit_l`
2. 降低输入影像分辨率
3. 使用 `--device cpu` 回退到 CPU
4. 训练时脚本会自动 OOM 降级，无需手动处理

</details>

<details>
<summary><b>Q: Python 3.14 安装 torch 报错 "from versions: none"</b></summary>

Python 3.14 尚无 CUDA 版 PyTorch wheel。请安装 Python 3.12：
```powershell
winget install Python.Python.3.12 --scope user
```
然后使用 `setup_env.ps1` 创建 Python 3.12 虚拟环境。

</details>

<details>
<summary><b>Q: PowerShell 执行 setup_env.ps1 报错</b></summary>

1. 确保使用 PowerShell 而非 cmd
2. 如遇执行策略限制：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. 脚本为纯 ASCII 英文，避免 GBK 编码问题

</details>

<details>
<summary><b>Q: 检测到的颗粒数量太少</b></summary>

1. 降低 `conf_threshold`（如改为 0.1）
2. 使用训练后的 `models/best.pt` 权重
3. 检查影像中小颗粒是否占据足够像素
4. 尝试 Fallback 模式：`python run_fallback.py`

</details>

<details>
<summary><b>Q: 分割掩码边界不准确</b></summary>

1. 使用 SAM vit_h 模型（精度最高）
2. 检查 SAM 模型权重文件是否完整
3. 调整 SAM 参数

</details>

<details>
<summary><b>Q: 中文图表显示为方框</b></summary>

```bash
# Ubuntu
sudo apt install fonts-wqy-microhei

# 或在代码中指定可用字体
# 修改 src/statistics/analyzer.py 中的 plt.rcParams["font.sans-serif"]
```

</details>

<details>
<summary><b>Q: 训练时 yolov8x 模型崩溃（loss=NaN）</b></summary>

数据集太小导致大模型过拟合后崩溃。解决方案：
1. 使用 yolov8n（`--stage quick`）替代 yolov8x
2. 扩充数据集（至少 50 张以上）
3. 降低学习率或减少 epochs

</details>

---

## 未来改进方向

- [ ] **SAM 2**：升级到 Meta 最新 SAM 2，支持视频流处理
- [x] **YOLOv8 训练模块**：已内置 `train_yolo.py` 两阶段训练脚本
- [ ] **多尺度推理**：使用 SAHI 框架对大尺寸影像进行切片推理
- [ ] **3D 粒径重建**：结合多视角无人机影像和 SfM 进行三维颗粒重建
- [ ] **Web 界面**：开发基于 Gradio 或 Streamlit 的 Web 前端
- [ ] **颗粒级配曲线**：自动生成级配曲线，与筛分试验结果对比
- [ ] **多类别识别**：区分不同岩性/风化程度的颗粒

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
