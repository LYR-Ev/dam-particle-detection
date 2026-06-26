"""
Fallback 检测脚本 — 基于 OpenCV 的颗粒检测方案。

当 YOLOv8 + SAM (需 torch/GPU/模型权重) 不可用时，使用 OpenCV 的
自适应阈值 + 分水岭算法进行颗粒检测，然后利用现有的测量和统计模块
完成分析。

用法:
    python run_fallback.py --input "D:/2026/.../images" --scale 0.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.measurement.particle_measurer import ParticleMeasurer, ParticleMeasurement
from src.statistics.analyzer import ParticleAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fallback")


def detect_particles_opencv(
    image_path: str,
    min_area: int = 200,
    max_area: int = 500000,
) -> List[dict]:
    """
    使用 OpenCV 自适应阈值 + 分水岭算法检测颗粒。

    处理流程:
    1. 灰度化 + 高斯模糊降噪
    2. 自适应阈值二值化
    3. 形态学操作 (开运算去噪 + 闭运算填孔)
    4. 距离变换 + 分水岭算法分割粘连颗粒
    5. 轮廓提取 + 面积过滤

    Args:
        image_path: 输入图像路径
        min_area:   最小颗粒面积 (像素)
        max_area:   最大颗粒面积 (像素)

    Returns:
        [{"mask": np.ndarray, "bbox": (x1,y1,x2,y2)}, ...]
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("无法读取图像: %s", image_path)
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(closed, kernel, iterations=5)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers.copy())
    img[markers == -1] = [0, 0, 255]

    results = []
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype(np.uint8)
        area = int(mask.sum())

        if area < min_area or area > max_area:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        results.append({
            "mask": mask,
            "bbox": (x, y, x + w, y + h),
        })

    logger.info("  检测到 %d 个颗粒", len(results))
    return results


def process_one_image(image_path: str, output_dir: Path, scale: float = None):
    """处理单张影像的完整流程"""
    image_name = Path(image_path).stem
    logger.info("处理: %s", image_name)

    particles = detect_particles_opencv(image_path)

    if len(particles) == 0:
        logger.warning("  未检测到颗粒，跳过")
        return None

    measurer = ParticleMeasurer(pixel_size_mm=scale)
    measurements = measurer.measure(particles)

    analyzer = ParticleAnalyzer(output_dir=str(output_dir))
    analyzer.analyze(measurements, image_name=image_name)

    img = cv2.imread(image_path)
    for i, m in enumerate(measurements):
        if m.mask is not None:
            color = np.random.randint(0, 255, (3,)).tolist()
            overlay = img.copy()
            overlay[m.mask > 0] = color
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

            x1, y1, x2, y2 = m.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(i), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    overlay_path = output_dir / f"{image_name}_overlay.png"
    cv2.imwrite(str(overlay_path), img)
    logger.info("  叠加图已保存: %s", overlay_path)

    return measurements


def process_all_images(input_dir: str, output_dir: str, scale: float = None):
    """批量处理所有影像并生成汇总报告"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_path.glob("*.png")) + \
                  sorted(input_path.glob("*.jpg")) + \
                  sorted(input_path.glob("*.jpeg"))

    if not image_files:
        logger.error("未找到图像文件 (.png/.jpg/.jpeg)")
        return

    logger.info("=" * 60)
    logger.info("堰塞坝颗粒智能检测 — 批量处理")
    logger.info("影像目录: %s", input_dir)
    logger.info("影像数量: %d", len(image_files))
    logger.info("比例尺: %s mm/pixel", scale if scale else "未设定 (像素单位)")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    all_measurements = []
    success = 0

    for img_file in image_files:
        try:
            measurements = process_one_image(str(img_file), output_path, scale)
            if measurements:
                all_measurements.extend(measurements)
                success += 1
        except Exception as e:
            logger.error("处理失败 %s: %s", img_file.name, e)

    logger.info("=" * 60)
    logger.info("处理完成: %d/%d 成功", success, len(image_files))
    logger.info("总计检测颗粒: %d 个", len(all_measurements))
    logger.info("=" * 60)

    if all_measurements:
        generate_summary_report(all_measurements, output_path, scale)

    logger.info("全部结果已保存至: %s", output_dir)


def generate_summary_report(measurements: List[ParticleMeasurement],
                            output_dir: Path, scale: float = None):
    """生成汇总统计报告"""
    from metrics import fit_rosin_rammler, compute_characteristic_sizes

    diameters = np.array([m.eq_circle_diameter_mm if m.eq_circle_diameter_mm > 0
                          else m.equivalent_diameter_mm for m in measurements])
    areas = np.array([m.area_mm2 for m in measurements])
    long_axes = np.array([m.long_axis_mm for m in measurements])
    short_axes = np.array([m.short_axis_mm for m in measurements])
    axis_ratios = np.array([m.axis_ratio for m in measurements])
    circularities = np.array([m.circularity for m in measurements])

    Xm, n, rr_r2 = fit_rosin_rammler(diameters)
    char_sizes = compute_characteristic_sizes(diameters)

    report_lines = []
    r = report_lines.append
    r("=" * 65)
    r("  堰塞坝表层颗粒物质智能检测分析 — 汇总报告")
    r("=" * 65)
    r(f"  检测颗粒总数: {len(measurements)}")
    r(f"  比例尺: {scale} mm/pixel" if scale else "  比例尺: 未设定 (像素单位)")
    r("")
    r("  【粒径统计】")
    r("  " + "-" * 40)
    r(f"  等效直径: 均值={diameters.mean():.2f}, 中位数={np.median(diameters):.2f}, "
      f"标准差={diameters.std():.2f}")
    r(f"  长轴:     均值={long_axes.mean():.2f}, 中位数={np.median(long_axes):.2f}, "
      f"标准差={long_axes.std():.2f}")
    r(f"  短轴:     均值={short_axes.mean():.2f}, 中位数={np.median(short_axes):.2f}, "
      f"标准差={short_axes.std():.2f}")
    r(f"  面积:     均值={areas.mean():.2f}, 中位数={np.median(areas):.2f}")
    r(f"  长宽比:   均值={axis_ratios.mean():.2f}, 标准差={axis_ratios.std():.2f}")
    r(f"  圆度:     均值={circularities.mean():.3f}, 标准差={circularities.std():.3f}")
    r("")
    r("  【Rosin-Rammler 分布拟合】")
    r("  " + "-" * 40)
    r(f"  特征粒径 Xm (X50): {Xm:.2f} mm")
    r(f"  均匀性指数 n:     {n:.4f}")
    r(f"  拟合 R²:          {rr_r2:.4f}")
    r("")
    r("  【特征粒径 X10~X100】")
    r("  " + "-" * 40)
    size_line = "  "
    for p in sorted(char_sizes.keys()):
        size_line += f"X{p}={char_sizes[p]:.1f}  "
    r(size_line)
    r("")
    r("=" * 65)

    report_text = "\n".join(report_lines)
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        print("\n[汇总报告已生成，见 outputs/summary_report.txt]")

    report_path = output_dir / "summary_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("汇总报告已保存: %s", report_path)

    analyzer = ParticleAnalyzer(output_dir=str(output_dir))
    analyzer.plot_particle_size_distribution(measurements, "summary")
    analyzer.plot_cumulative_passing_with_rr(measurements, "summary")

    import pandas as pd
    records = []
    for m in measurements:
        records.append({
            "particle_id": m.particle_id,
            "area_mm2": round(m.area_mm2, 2),
            "long_axis_mm": round(m.long_axis_mm, 2),
            "short_axis_mm": round(m.short_axis_mm, 2),
            "equivalent_diameter_mm": round(m.equivalent_diameter_mm, 2),
            "eq_circle_diameter_mm": round(m.eq_circle_diameter_mm, 2),
            "axis_ratio": round(m.axis_ratio, 3),
            "circularity": round(m.circularity, 3),
            "perimeter_mm": round(m.perimeter_mm, 2),
            "orientation_deg": round(m.orientation_deg, 1),
        })
    df = pd.DataFrame(records)
    csv_path = output_dir / "summary_all_particles.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("汇总 CSV 已保存: %s (%d 条)", csv_path, len(df))


def main():
    parser = argparse.ArgumentParser(description="Fallback 颗粒检测 (基于 OpenCV)")
    parser.add_argument("--input", "-i", required=True,
                        help="输入影像目录路径")
    parser.add_argument("--output", "-o", default="outputs",
                        help="输出目录 (默认: outputs)")
    parser.add_argument("--scale", "-s", type=float, default=None,
                        help="像素分辨率 mm/pixel")
    parser.add_argument("--min-area", type=int, default=200,
                        help="最小颗粒面积 (像素, 默认: 200)")
    parser.add_argument("--max-area", type=int, default=500000,
                        help="最大颗粒面积 (像素, 默认: 500000)")
    args = parser.parse_args()

    process_all_images(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()