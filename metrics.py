"""
基于2024年《Minerals》顶刊论文《Identification of Rock Fragments after Blasting
by Using Deep Learning-Based Segment Anything Model》的评估指标体系。

实现了论文中所有的像素级和粒径级评估指标，严格遵循论文公式(1)-(9)。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ============================================================
# 第一部分：像素级评估指标 — 论文公式(1)~(3)
# ============================================================

def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算像素精度 (Pixel Accuracy, PA) — 论文公式(1)

    PA = Σ p_ii / Σ Σ p_ij
    即正确分类的像素数占总像素数的比例。

    Args:
        pred_mask: 预测的二值掩码 (H, W)，值为 0 或 1
        gt_mask:   真值的二值掩码 (H, W)，值为 0 或 1

    Returns:
        PA 值，范围 [0, 1]
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))
    total = pred_bin.size
    if total == 0:
        return 0.0
    return float(tp + tn) / total


def mean_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """
    计算平均交并比 (mean Intersection over Union, mIOU) — 论文公式(2)

    mIOU = (1/(n+1)) * Σ (p_ii / (Σ_j p_ij + Σ_j p_ji - p_ii))
    对每个前景类别和背景类别分别计算 IoU 后取平均。

    此处简化为：对每个颗粒实例计算 IOU，然后对所有实例取平均。

    Args:
        pred_masks: 预测掩码列表
        gt_masks:   真值掩码列表

    Returns:
        mIOU 值，范围 [0, 1]
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    iou_values = []
    for pm, gm in zip(pred_masks, gt_masks):
        iou = compute_iou(pm, gm)
        iou_values.append(iou)

    return float(np.mean(iou_values)) if iou_values else 0.0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算单个掩码的交并比 (IoU)。

    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        pred_mask: 预测的二值掩码
        gt_mask:   真值的二值掩码

    Returns:
        IoU 值，范围 [0, 1]
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 0.0
    return float(intersection) / union


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算 Dice 系数 — 论文公式(3)

    Dice = 2|X ∩ Y| / (|X| + |Y|)

    用于衡量预测掩码与真值掩码之间的重叠程度。

    Args:
        pred_mask: 预测的二值掩码
        gt_mask:   真值的二值掩码

    Returns:
        Dice 系数，范围 [0, 1]
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = 2.0 * np.sum(pred_bin & gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total == 0:
        return 0.0
    return float(intersection) / total


def compute_all_pixel_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    一次性计算所有像素级评估指标。

    Args:
        pred_masks: 预测掩码列表
        gt_masks:   真值掩码列表

    Returns:
        包含 PA, mIOU, Dice 的字典
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return {"PA": 0.0, "mIOU": 0.0, "Dice": 0.0, "avg_IoU": 0.0}

    pa_values = []
    iou_values = []
    dice_values = []

    for pm, gm in zip(pred_masks, gt_masks):
        pa_values.append(pixel_accuracy(pm, gm))
        iou_values.append(compute_iou(pm, gm))
        dice_values.append(dice_coefficient(pm, gm))

    return {
        "PA": float(np.mean(pa_values)),
        "mIOU": float(np.mean(iou_values)),
        "Dice": float(np.mean(dice_values)),
        "avg_IoU": float(np.mean(iou_values)),
    }


# ============================================================
# 第二部分：等效圆直径 — 论文公式(4)
# ============================================================

def equivalent_circle_diameter(area_mm2: np.ndarray) -> np.ndarray:
    """
    计算等效圆直径 (Equivalent Circle Diameter) — 论文公式(4)

    D = 2 * sqrt(S / π)

    其中 S 为颗粒的投影面积 (mm²)，D 为与颗粒投影面积相等的圆的直径 (mm)。

    Args:
        area_mm2: 颗粒面积数组 (mm²)

    Returns:
        等效圆直径数组 (mm)
    """
    area = np.asarray(area_mm2, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        result = 2.0 * np.sqrt(area / np.pi)
    result = np.nan_to_num(result, nan=0.0)
    return result


def circularity(area_mm2: np.ndarray, perimeter_mm: np.ndarray) -> np.ndarray:
    """
    计算颗粒圆度 (Circularity)。

    Circularity = 4π × Area / Perimeter²

    正圆的圆度为 1，越不规则的形状圆度越小。

    Args:
        area_mm2:      颗粒面积 (mm²)
        perimeter_mm:  颗粒周长 (mm)

    Returns:
        圆度数组，范围 (0, 1]
    """
    area = np.asarray(area_mm2, dtype=np.float64)
    perimeter = np.asarray(perimeter_mm, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = 4.0 * np.pi * area / (perimeter ** 2)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0)
    return np.clip(result, 0.0, 1.0)


# ============================================================
# 第三部分：Rosin-Rammler 粒径分布 — 论文公式(5)~(9)
# ============================================================

def rosin_rammler_cdf(X: np.ndarray, Xm: float, n: float) -> np.ndarray:
    """
    Rosin-Rammler 累积分布函数 — 论文公式(8)

    R(x) = 1 - exp(-0.693 × (X / Xm)^n)

    其中：
      - Xm: 特征粒径 (中值粒径 X50)，即累积通过率为 50% 时的粒径
      - n:  均匀性指数，n 越大表示粒径分布越均匀

    Args:
        X:   粒径值数组 (mm)
        Xm:  特征粒径 X50 (mm)
        n:   均匀性指数

    Returns:
        累积通过率数组，范围 [0, 1]
    """
    ratio = np.asarray(X, dtype=np.float64) / Xm
    with np.errstate(over="ignore"):
        result = 1.0 - np.exp(-0.693 * (ratio ** n))
    result = np.nan_to_num(result, nan=0.0, posinf=1.0)
    return np.clip(result, 0.0, 1.0)


def rosin_rammler_pdf(X: np.ndarray, Xm: float, n: float) -> np.ndarray:
    """
    Rosin-Rammler 概率密度函数 — 论文公式(8)的导数

    f(x) = (0.693 × n / Xm) × (X / Xm)^(n-1) × exp(-0.693 × (X / Xm)^n)

    Args:
        X:   粒径值数组 (mm)
        Xm:  特征粒径 (mm)
        n:   均匀性指数

    Returns:
        概率密度值数组
    """
    ratio = np.asarray(X, dtype=np.float64) / Xm
    result = (0.693 * n / Xm) * (ratio ** (n - 1)) * np.exp(-0.693 * (ratio ** n))
    result = np.nan_to_num(result, nan=0.0)
    return result


def fit_rosin_rammler(diameters_mm: np.ndarray) -> Tuple[float, float, float]:
    """
    使用非线性最小二乘法拟合 Rosin-Rammler 分布参数 — 论文公式(9)

    公式(9): n = 0.842 / (ln(k80) - ln(k50))

    其中 k80 = X80/X50, k50 = 1

    然后使用 curve_fit 精修 n 和 Xm 参数。

    Args:
        diameters_mm: 所有颗粒的等效直径数组 (mm)

    Returns:
        (Xm, n, R²) — 特征粒径、均匀性指数、拟合决定系数
    """
    if len(diameters_mm) < 5:
        logger.warning("粒径数据不足 (n=%d)，无法拟合 RR 分布", len(diameters_mm))
        return (0.0, 1.0, 0.0)

    sorted_d = np.sort(diameters_mm)
    n_total = len(sorted_d)
    cumulative = np.arange(1, n_total + 1) / n_total

    X50 = float(np.median(diameters_mm))
    X80 = float(np.percentile(diameters_mm, 80))

    if X50 <= 0:
        return (0.0, 1.0, 0.0)

    # 初始 n 值 — 论文公式(9)
    if X80 > X50:
        n_init = 0.842 / (np.log(X80) - np.log(X50))
        n_init = max(0.3, min(n_init, 10.0))
    else:
        n_init = 1.5

    def _rr_func(X, Xm, n):
        return rosin_rammler_cdf(X, Xm, n)

    try:
        popt, pcov = curve_fit(
            _rr_func, sorted_d, cumulative,
            p0=[X50, n_init],
            bounds=([1e-6, 0.1], [1e6, 20.0]),
            maxfev=10000,
        )
        Xm_fit, n_fit = popt[0], popt[1]
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("RR 拟合失败: %s，使用初始估计值", e)
        Xm_fit, n_fit = X50, n_init

    predicted = rosin_rammler_cdf(sorted_d, Xm_fit, n_fit)
    ss_res = np.sum((cumulative - predicted) ** 2)
    ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return (float(Xm_fit), float(n_fit), float(max(r_squared, 0.0)))


def compute_characteristic_sizes(
    diameters_mm: np.ndarray,
    percentiles: List[int] = None,
) -> Dict[int, float]:
    """
    计算特征粒径 X10, X20, ..., X100 — 论文 Table 2/Table 4 格式

    Args:
        diameters_mm: 所有颗粒的等效直径数组 (mm)
        percentiles:  需要计算的百分位列表，默认 X10~X100 (步长 10)

    Returns:
        {10: X10值, 20: X20值, ..., 100: X100值}
    """
    if percentiles is None:
        percentiles = list(range(10, 101, 10))

    if len(diameters_mm) == 0:
        return {p: 0.0 for p in percentiles}

    result = {}
    for p in percentiles:
        result[p] = float(np.percentile(diameters_mm, p))
    return result


def calc_rr_characteristic_sizes(
    Xm: float, n: float, percentiles: List[int] = None
) -> Dict[int, float]:
    """
    根据 Rosin-Rammler 拟合参数反推特征粒径。

    由公式(8)反解: X = Xm × (-ln(1-R) / 0.693)^(1/n)

    Args:
        Xm:          特征粒径 X50 (mm)
        n:           均匀性指数
        percentiles:  百分位列表

    Returns:
        {10: X10_RR, 20: X20_RR, ...}
    """
    if percentiles is None:
        percentiles = list(range(10, 101, 10))

    if Xm <= 0 or n <= 0:
        return {p: 0.0 for p in percentiles}

    result = {}
    for p in percentiles:
        R = p / 100.0
        if R >= 1.0:
            result[p] = Xm * 5.0
        elif R <= 0.0:
            result[p] = 0.0
        else:
            result[p] = float(Xm * ((-np.log(1.0 - R) / 0.693) ** (1.0 / n)))
    return result


# ============================================================
# 第四部分：粒径级评估指标 — MAE, RMSE, R²
# ============================================================

def compute_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)

    MAE = (1/n) × Σ |pred_i - gt_i|

    Args:
        predicted:   预测值数组
        ground_truth: 真值数组

    Returns:
        MAE 值
    """
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(ground_truth))))


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Square Error)

    RMSE = sqrt((1/n) × Σ (pred_i - gt_i)²)

    Args:
        predicted:   预测值数组
        ground_truth: 真值数组

    Returns:
        RMSE 值
    """
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(ground_truth)) ** 2)))


def compute_r_squared(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    计算决定系数 (Coefficient of Determination, R²)

    R² = 1 - Σ(pred_i - gt_i)² / Σ(gt_i - mean(gt))²

    Args:
        predicted:   预测值数组
        ground_truth: 真值数组

    Returns:
        R² 值，范围 (-∞, 1]，接近 1 表示拟合越好
    """
    pred = np.asarray(predicted)
    gt = np.asarray(ground_truth)
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_mre(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    计算平均相对误差 (Mean Relative Error, MRE)

    MRE = (1/n) × Σ |pred_i - gt_i| / gt_i

    论文 Figure 15 中使用。
    """
    pred = np.asarray(predicted)
    gt = np.asarray(ground_truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(pred - gt) / gt
    relative = np.nan_to_num(relative, nan=0.0, posinf=0.0)
    return float(np.mean(relative))


def compute_slope_intercept(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> Tuple[float, float, float]:
    """
    计算预测值与真值的线性回归参数 — 论文 Figure 15 格式。

    gt = slope × pred + intercept

    Args:
        predicted:   预测值数组
        ground_truth: 真值数组

    Returns:
        (slope, intercept, r_value)
    """
    try:
        result = linregress(predicted, ground_truth)
        return (float(result.slope), float(result.intercept), float(result.rvalue))
    except (ValueError, TypeError) as e:
        logger.warning("线性回归失败: %s", e)
        return (0.0, 0.0, 0.0)


# ============================================================
# 第五部分：完整评估报告
# ============================================================

@dataclass
class EvaluationReport:
    """完整的评估报告数据类 — 对应论文 Table 1, 2, 3, 4"""

    image_name: str = ""
    total_particles: int = 0

    # 像素级指标 — 对应 Table 1 / Table 3
    PA: float = 0.0
    mIOU: float = 0.0
    Dice: float = 0.0

    # Rosin-Rammler 拟合参数
    RR_Xm: float = 0.0
    RR_n: float = 0.0
    RR_R2: float = 0.0

    # 特征粒径 — 对应 Table 2 / Table 4
    pred_char_sizes: Dict[int, float] = field(default_factory=dict)
    gt_char_sizes: Dict[int, float] = field(default_factory=dict)

    # 粒径级评估指标 — 对应 Figure 15
    MAE_char: float = 0.0
    RMSE_char: float = 0.0
    R2_char: float = 0.0
    MRE_char: float = 0.0

    # 回归参数
    slope: float = 0.0
    intercept: float = 0.0
    r_value: float = 0.0

    # 颗粒级配数据
    pred_diameters: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    gt_diameters: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def generate_evaluation_report(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_measurements: List,
    gt_measurements: List,
    image_name: str = "test_image",
    scale_mm_per_pixel: Optional[float] = None,
) -> EvaluationReport:
    """
    一键生成完整评估报告 — 输出格式与论文 Table 1, 3, 4 完全一致。

    功能：
    1. 计算 PA / mIOU / Dice（像素级评估）
    2. 计算等效圆直径并拟合 Rosin-Rammler 分布
    3. 计算特征粒径 X10~X100 并与真值对比
    4. 计算 MAE / RMSE / R² / MRE（粒径级评估）

    Args:
        pred_masks:        预测分割掩码列表
        gt_masks:          真值分割掩码列表
        pred_measurements: 预测的颗粒测量结果 (ParticleMeasurement 列表)
        gt_measurements:   真值的颗粒测量结果 (ParticleMeasurement 列表)
        image_name:        图像名称
        scale_mm_per_pixel: 像素-毫米比例尺

    Returns:
        EvaluationReport 评估报告对象
    """
    report = EvaluationReport(image_name=image_name)

    # 像素级评估
    pixel_metrics = compute_all_pixel_metrics(pred_masks, gt_masks)
    report.PA = pixel_metrics["PA"]
    report.mIOU = pixel_metrics["mIOU"]
    report.Dice = pixel_metrics["Dice"]

    # 提取等效直径
    pred_diams = np.array([m.equivalent_diameter_mm for m in pred_measurements])
    gt_diams = np.array([m.equivalent_diameter_mm for m in gt_measurements])

    report.pred_diameters = pred_diams
    report.gt_diameters = gt_diams
    report.total_particles = len(pred_measurements)

    # Rosin-Rammler 拟合
    if len(pred_diams) >= 5:
        Xm, n, rr_r2 = fit_rosin_rammler(pred_diams)
        report.RR_Xm = Xm
        report.RR_n = n
        report.RR_R2 = rr_r2

    # 特征粒径计算
    percentiles = list(range(10, 101, 10))
    report.pred_char_sizes = compute_characteristic_sizes(pred_diams, percentiles)
    report.gt_char_sizes = compute_characteristic_sizes(gt_diams, percentiles)

    # 粒径级评估 — 对比特征粒径
    pred_chars = np.array([report.pred_char_sizes[p] for p in percentiles])
    gt_chars = np.array([report.gt_char_sizes[p] for p in percentiles])

    if len(gt_chars) > 0 and np.sum(gt_chars) > 0:
        report.MAE_char = compute_mae(pred_chars, gt_chars)
        report.RMSE_char = compute_rmse(pred_chars, gt_chars)
        report.R2_char = compute_r_squared(pred_chars, gt_chars)
        report.MRE_char = compute_mre(pred_chars, gt_chars)
        report.slope, report.intercept, report.r_value = compute_slope_intercept(
            pred_chars, gt_chars
        )

    return report


def print_evaluation_report(report: EvaluationReport) -> str:
    """
    将评估报告格式化为可打印/可保存的字符串 — 格式与论文 Table 1/3/4 一致。

    Args:
        report: EvaluationReport 对象

    Returns:
        格式化的评估报告字符串
    """
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  堰塞坝表层颗粒物质智能检测 — 评估报告")
    lines.append(f"  依据论文: Minerals 2024, 14, 654")
    lines.append(f"  图像: {report.image_name}")
    lines.append(sep)

    # Table 1 / Table 3 格式 — 像素级评估
    lines.append("")
    lines.append("  【Table 1/3 格式】像素级分割性能评估")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'指标':<20} {'数值':>10}")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'PA (Pixel Accuracy)':<20} {report.PA:>10.4f}")
    lines.append(f"  {'mIOU (mean IoU)':<20} {report.mIOU:>10.4f}")
    lines.append(f"  {'Dice Coefficient':<20} {report.Dice:>10.4f}")
    lines.append("  " + "-" * 50)

    # Rosin-Rammler 拟合参数
    lines.append("")
    lines.append("  【Table 2 格式】Rosin-Rammler 分布拟合参数")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'特征粒径 Xm (X50, mm)':<30} {report.RR_Xm:>10.2f}")
    lines.append(f"  {'均匀性指数 n':<30} {report.RR_n:>10.4f}")
    lines.append(f"  {'拟合 R²':<30} {report.RR_R2:>10.4f}")
    lines.append("  " + "-" * 50)

    # Table 4 格式 — 特征粒径对比
    lines.append("")
    lines.append("  【Table 4 格式】10 个特征粒径对比 (mm)")
    lines.append("  " + "-" * 70)
    header = f"  {'特征粒径':<12} "
    for p in sorted(report.pred_char_sizes.keys()):
        header += f"{'X'+str(p):>8}"
    lines.append(header)
    lines.append("  " + "-" * 70)

    gt_row = f"  {'真值 (mm)':<12} "
    for p in sorted(report.gt_char_sizes.keys()):
        gt_row += f"{report.gt_char_sizes[p]:>8.2f}"
    lines.append(gt_row)

    pred_row = f"  {'预测值 (mm)':<12} "
    for p in sorted(report.pred_char_sizes.keys()):
        pred_row += f"{report.pred_char_sizes[p]:>8.2f}"
    lines.append(pred_row)

    diff_row = f"  {'差值':<12} "
    for p in sorted(report.pred_char_sizes.keys()):
        diff = report.pred_char_sizes[p] - report.gt_char_sizes[p]
        diff_row += f"{diff:>8.2f}"
    lines.append(diff_row)
    lines.append("  " + "-" * 70)

    # 粒径级评估指标
    lines.append("")
    lines.append("  【Figure 15 格式】粒径级评估指标")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'MAE (mm)':<25} {report.MAE_char:>10.4f}")
    lines.append(f"  {'RMSE (mm)':<25} {report.RMSE_char:>10.4f}")
    lines.append(f"  {'R²':<25} {report.R2_char:>10.4f}")
    lines.append(f"  {'MRE':<25} {report.MRE_char:>10.4f}")
    lines.append(f"  {'回归斜率 (Slope)':<25} {report.slope:>10.4f}")
    lines.append(f"  {'回归截距 (Intercept)':<25} {report.intercept:>10.4f}")
    lines.append(f"  {'相关系数 (r)':<25} {report.r_value:>10.4f}")
    lines.append("  " + "-" * 50)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def export_evaluation_csv(report: EvaluationReport, output_path: str):
    """
    将评估报告导出为 CSV 文件。

    输出三个表格：
    - {prefix}_pixel_metrics.csv — 像素级指标
    - {prefix}_char_sizes.csv    — 特征粒径对比
    - {prefix}_grain_level.csv   — 粒径级指标 + RR 参数

    Args:
        report:      EvaluationReport 对象
        output_path: 输出基础路径 (会自动添加后缀)
    """
    import pandas as pd

    base = Path(output_path).parent / Path(output_path).stem

    # Table 1/3 — 像素级指标
    pixel_df = pd.DataFrame([
        {"指标": "PA (Pixel Accuracy)", "数值": round(report.PA, 4)},
        {"指标": "mIOU (mean IoU)", "数值": round(report.mIOU, 4)},
        {"指标": "Dice Coefficient", "数值": round(report.Dice, 4)},
    ])
    pixel_df.to_csv(f"{base}_pixel_metrics.csv", index=False, encoding="utf-8-sig")

    # Table 4 — 特征粒径
    chars_data = []
    for p in sorted(report.pred_char_sizes.keys()):
        chars_data.append({
            "特征粒径": f"X{p}",
            "真值_mm": round(report.gt_char_sizes.get(p, 0), 2),
            "预测值_mm": round(report.pred_char_sizes.get(p, 0), 2),
            "差值_mm": round(report.pred_char_sizes.get(p, 0) - report.gt_char_sizes.get(p, 0), 2),
        })
    chars_df = pd.DataFrame(chars_data)
    chars_df.to_csv(f"{base}_char_sizes.csv", index=False, encoding="utf-8-sig")

    # Figure 15 — 粒径级 + RR
    grain_df = pd.DataFrame([
        {"指标": "RR_Xm_mm", "数值": round(report.RR_Xm, 4)},
        {"指标": "RR_n", "数值": round(report.RR_n, 4)},
        {"指标": "RR_R2", "数值": round(report.RR_R2, 4)},
        {"指标": "MAE_mm", "数值": round(report.MAE_char, 4)},
        {"指标": "RMSE_mm", "数值": round(report.RMSE_char, 4)},
        {"指标": "R2", "数值": round(report.R2_char, 4)},
        {"指标": "MRE", "数值": round(report.MRE_char, 4)},
        {"指标": "Slope", "数值": round(report.slope, 4)},
        {"指标": "Intercept", "数值": round(report.intercept, 4)},
        {"指标": "r_value", "数值": round(report.r_value, 4)},
        {"指标": "total_particles", "数值": report.total_particles},
    ])
    grain_df.to_csv(f"{base}_grain_level.csv", index=False, encoding="utf-8-sig")

    logger.info("评估报告 CSV 已导出至: %s_*.csv", base)