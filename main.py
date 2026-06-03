#!/usr/bin/env python3
"""
堰塞坝表层颗粒物质智能检测分析系统 - 入口脚本。

基于 YOLOv8 + SAM 的组合方案，对无人机拍摄的堰塞坝 RGB 影像
进行颗粒检测、分割、粒径测量与统计分析。

支持两种模式：
  1. 检测模式：影像 → 颗粒检测 → 分割 → 粒径 → 统计报告
  2. 评估模式：将预测结果与标注真值对比，生成论文格式的评估报告

用法:
    # 单张影像处理
    python main.py --image path/to/image.jpg

    # 指定像素分辨率 (mm/pixel)
    python main.py --image path/to/image.jpg --scale 0.5

    # 批量处理
    python main.py --dir path/to/images/ --scale 0.5

    # 评估模式
    python main.py --evaluate --image path/to/image.jpg \\
        --ground-truth path/to/gt/ --output-report outputs/

    # 使用自定义配置
    python main.py --config config/config.yaml --image path/to/image.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import DamParticlePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def load_ground_truth_masks(gt_dir: str, image_name: str):
    """
    从标注目录加载真值掩码。

    支持两种格式：
    1. PNG 掩码图 — 每个像素值代表一个颗粒实例 ID
    2. 单独的 mask PNG 文件 — 文件名 pattern: {image_name}_mask_{i}.png

    Args:
        gt_dir:     标注数据根目录
        image_name: 图像名称 (不含扩展名)

    Returns:
        gt_masks 列表 (List[np.ndarray])
    """
    import cv2
    import numpy as np

    gt_path = Path(gt_dir)

    instance_mask_path = gt_path / f"{image_name}_mask.png"
    if instance_mask_path.exists():
        mask_img = cv2.imread(str(instance_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise ValueError(f"无法读取标注掩码: {instance_mask_path}")

        gt_masks = []
        instance_ids = np.unique(mask_img)
        for inst_id in instance_ids:
            if inst_id == 0:
                continue
            gt_masks.append((mask_img == inst_id).astype(np.uint8))
        logger.info("从实例掩码加载 %d 个真值掩码", len(gt_masks))
        return gt_masks

    mask_files = sorted(gt_path.glob(f"{image_name}_mask_*.png"))
    if mask_files:
        gt_masks = []
        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                gt_masks.append((mask > 0).astype(np.uint8))
        logger.info("从单独掩码文件加载 %d 个真值掩码", len(gt_masks))
        return gt_masks

    raise FileNotFoundError(
        f"未找到标注数据: {gt_dir}。"
        f"请确保存在 {image_name}_mask.png (实例掩码) "
        f"或 {image_name}_mask_0.png, {image_name}_mask_1.png 等单独掩码文件。"
    )


def run_evaluation_mode(args):
    """
    评估模式主流程：

    1. 运行完整 pipeline，生成预测结果
    2. 加载真值标注数据（掩码 + 粒径真值 CSV）
    3. 调用 metrics.py 计算所有评估指标
    4. 生成论文格式的评估报告（文本 + CSV + 图表）

    Args:
        args: 命令行参数
    """
    from metrics import (
        generate_evaluation_report,
        print_evaluation_report,
        export_evaluation_csv,
    )

    pipeline = DamParticlePipeline(args.config)
    image_path = Path(args.image)
    image_name = image_path.stem

    if args.device:
        pipeline.yolo_detector.device = args.device
        pipeline.sam_segmentor.device = args.device

    report_dir = Path(args.output_report) if args.output_report else Path("outputs")
    report_dir.mkdir(parents=True, exist_ok=True)
    pipeline.output_dir = report_dir
    pipeline.analyzer.output_dir = report_dir

    logger.info("=" * 60)
    logger.info("评估模式: %s", image_name)
    logger.info("=" * 60)

    result = pipeline.run(
        str(image_path),
        image_name=image_name,
        scale_mm_per_pixel=args.scale,
    )

    pred_measurements = result.get("measurements", [])
    if len(pred_measurements) == 0:
        logger.error("未检测到任何颗粒，无法进行评估")
        return

    try:
        gt_masks = load_ground_truth_masks(args.ground_truth, image_name)
    except FileNotFoundError as e:
        logger.error("加载标注数据失败: %s", e)
        return

    pred_masks = [m.mask for m in pred_measurements if m.mask is not None]

    gt_measurements = _load_gt_measurements(
        args.ground_truth, image_name, pipeline.measurer
    )

    min_len = min(len(pred_masks), len(gt_masks))
    if min_len == 0:
        logger.error("无有效掩码对比")
        return
    pred_masks = pred_masks[:min_len]
    gt_masks = gt_masks[:min_len]

    report = generate_evaluation_report(
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        pred_measurements=pred_measurements,
        gt_measurements=gt_measurements,
        image_name=image_name,
        scale_mm_per_pixel=args.scale,
    )

    report_text = print_evaluation_report(report)
    print("\n" + report_text)

    pipeline.analyzer.save_evaluation_report(report_text, image_name)

    csv_base = str(report_dir / f"{image_name}_evaluation")
    export_evaluation_csv(report, csv_base)

    pipeline.analyzer.plot_particle_size_distribution(pred_measurements, image_name)
    pipeline.analyzer.plot_cumulative_passing_with_rr(pred_measurements, image_name)
    pipeline.analyzer.plot_characteristic_sizes_comparison(
        report.pred_char_sizes, report.gt_char_sizes, image_name
    )
    pipeline.analyzer.plot_evaluation_metrics_bars(
        {
            "PA": report.PA,
            "mIOU": report.mIOU,
            "Dice": report.Dice,
            "R² (Char)": report.R2_char,
            "RR R²": report.RR_R2,
        },
        image_name,
        title="评估指标总览 (Evaluation Metrics Overview)",
    )

    logger.info("评估完成! 报告保存至: %s", report_dir)


def _load_gt_measurements(gt_dir: str, image_name: str, measurer):
    """
    加载真值粒径测量数据。

    如果存在 {image_name}_particles.csv，直接读取；
    否则从真值掩码重新计算粒径。

    Args:
        gt_dir:   标注数据目录
        image_name: 图像名称
        measurer:  ParticleMeasurer 实例

    Returns:
        ParticleMeasurement 列表
    """
    import pandas as pd

    gt_path = Path(gt_dir)
    csv_path = gt_path / f"{image_name}_particles.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        logger.info("从 CSV 加载 %d 条真值粒径数据", len(df))
        return _csv_to_measurements(df)

    logger.info("真值 CSV 不存在，将从掩码重新计算粒径")
    gt_masks = load_ground_truth_masks(gt_dir, image_name)
    seg_results = [{"mask": m, "bbox": (0, 0, m.shape[1], m.shape[0])} for m in gt_masks]
    return measurer.measure(seg_results)


def _csv_to_measurements(df):
    """将 CSV DataFrame 转换为 ParticleMeasurement 列表"""
    from src.measurement.particle_measurer import ParticleMeasurement

    measurements = []
    for _, row in df.iterrows():
        m = ParticleMeasurement(
            particle_id=int(row.get("particle_id", len(measurements))),
            bbox=(row.get("x1", 0), row.get("y1", 0),
                  row.get("x2", 0), row.get("y2", 0)),
            area_pixels=0,
            area_mm2=float(row.get("area_mm2", 0)),
            long_axis_pixels=0,
            short_axis_pixels=0,
            long_axis_mm=float(row.get("long_axis_mm", 0)),
            short_axis_mm=float(row.get("short_axis_mm", 0)),
            equivalent_diameter_pixels=0,
            equivalent_diameter_mm=float(row.get("equivalent_diameter_mm", 0)),
            axis_ratio=float(row.get("axis_ratio", 1.0)),
            centroid=(row.get("centroid_x", 0), row.get("centroid_y", 0)),
            orientation_deg=float(row.get("orientation_deg", 0)),
            perimeter_pixels=0,
            perimeter_mm=float(row.get("perimeter_mm", 0)) if "perimeter_mm" in row else 0,
            eq_circle_diameter_mm=float(row.get("eq_circle_diameter_mm", 0)) if "eq_circle_diameter_mm" in row else 0,
            circularity=float(row.get("circularity", 0)) if "circularity" in row else 0,
        )
        measurements.append(m)
    return measurements


def main():
    parser = argparse.ArgumentParser(
        description="堰塞坝表层颗粒物质智能检测分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --image dam_uav_001.jpg
  python main.py --image dam_uav_001.jpg --scale 0.35
  python main.py --dir ./uav_images/ --scale 0.35
  python main.py --evaluate --image dam.jpg --ground-truth ./gt/ --output-report ./reports/
  python main.py --config config/custom.yaml --image dam.jpg
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i", type=str,
        help="输入影像路径"
    )
    input_group.add_argument(
        "--dir", "-d", type=str,
        help="批量处理的影像目录路径"
    )

    parser.add_argument(
        "--config", "-c", type=str, default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)"
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=None,
        help="像素分辨率 mm/pixel (覆盖配置文件中的设置)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出目录 (默认: outputs/)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="计算设备 (覆盖配置文件中的设置)"
    )

    parser.add_argument(
        "--evaluate", action="store_true",
        help="进入评估模式，将预测结果与标注真值对比，生成论文格式评估报告"
    )
    parser.add_argument(
        "--ground-truth", type=str, default="./ground_truth/",
        help="标注数据文件夹路径 (评估模式必需)"
    )
    parser.add_argument(
        "--output-report", type=str, default=None,
        help="评估报告输出路径 (默认: outputs/)"
    )

    args = parser.parse_args()

    if args.evaluate:
        if args.dir:
            parser.error("评估模式暂不支持批量处理，请使用 --image 指定单张影像")
        if not args.image:
            parser.error("评估模式需要指定 --image")
        if not args.ground_truth:
            parser.error("评估模式需要指定 --ground-truth")
        run_evaluation_mode(args)
        return

    pipeline = DamParticlePipeline(args.config)

    if args.device:
        pipeline.yolo_detector.device = args.device
        pipeline.sam_segmentor.device = args.device

    if args.output:
        pipeline.output_dir = Path(args.output)
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.analyzer.output_dir = pipeline.output_dir

    if args.image:
        pipeline.run(
            args.image,
            scale_mm_per_pixel=args.scale,
        )
    elif args.dir:
        pipeline.run_batch(
            args.dir,
            scale_mm_per_pixel=args.scale,
        )


if __name__ == "__main__":
    main()