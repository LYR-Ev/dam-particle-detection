import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.measurement.particle_measurer import ParticleMeasurement

logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class StatisticalReport:
    """粒径分布统计报告。"""

    total_particles: int
    long_axis_mm: dict
    short_axis_mm: dict
    equivalent_diameter_mm: dict
    area_mm2: dict

    @classmethod
    def from_measurements(
        cls, measurements: List[ParticleMeasurement]
    ) -> "StatisticalReport":
        """从测量结果列表生成统计报告。"""
        long_axes = np.array([m.long_axis_mm for m in measurements])
        short_axes = np.array([m.short_axis_mm for m in measurements])
        eq_diams = np.array([m.equivalent_diameter_mm for m in measurements])
        areas = np.array([m.area_mm2 for m in measurements])

        return cls(
            total_particles=len(measurements),
            long_axis_mm=cls._compute_stats(long_axes),
            short_axis_mm=cls._compute_stats(short_axes),
            equivalent_diameter_mm=cls._compute_stats(eq_diams),
            area_mm2=cls._compute_stats(areas),
        )

    @staticmethod
    def _compute_stats(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {}
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "d10": float(np.percentile(values, 10)),
            "d25": float(np.percentile(values, 25)),
            "d50": float(np.percentile(values, 50)),
            "d75": float(np.percentile(values, 75)),
            "d90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }


class ParticleAnalyzer:
    """颗粒粒径分布统计分析与可视化。

    生成直方图、累计分布曲线、统计报告 CSV 等。
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        bin_count: int = 30,
        save_masks: bool = True,
        save_overlay: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bin_count = bin_count
        self.save_masks = save_masks
        self.save_overlay = save_overlay

    def analyze(
        self,
        measurements: List[ParticleMeasurement],
        image: Optional[np.ndarray] = None,
        image_name: str = "result",
    ) -> StatisticalReport:
        """执行完整的粒径统计分析。

        Args:
            measurements: 颗粒测量结果列表。
            image: 原始图像 (可选，用于可视化叠加)。
            image_name: 输出文件前缀。

        Returns:
            StatisticalReport 统计报告对象。
        """
        if len(measurements) == 0:
            logger.warning("没有有效的颗粒测量数据，跳过分析")
            return StatisticalReport(
                total_particles=0,
                long_axis_mm={},
                short_axis_mm={},
                equivalent_diameter_mm={},
                area_mm2={},
            )

        report = StatisticalReport.from_measurements(measurements)

        self._plot_distribution_histograms(measurements, image_name)
        self._plot_cumulative_distribution(measurements, image_name)
        self._plot_summary_dashboard(measurements, image_name)
        self._export_csv(measurements, report, image_name)
        self._save_report_txt(report, image_name)

        if image is not None and self.save_overlay:
            self._save_overlay_image(image, measurements, image_name)

        logger.info("统计分析完成，结果保存至: %s", self.output_dir)
        return report

    def _plot_distribution_histograms(
        self, measurements: List[ParticleMeasurement], name: str
    ):
        """绘制长轴、短轴、等效直径的直方图。"""
        long_axes = [m.long_axis_mm for m in measurements]
        short_axes = [m.short_axis_mm for m in measurements]
        eq_diams = [m.equivalent_diameter_mm for m in measurements]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        titles = [
            ("颗粒长轴分布", "long_axis"),
            ("颗粒短轴分布", "short_axis"),
            ("等效直径分布", "equivalent_diameter"),
        ]
        data_map = {
            "long_axis": long_axes,
            "short_axis": short_axes,
            "equivalent_diameter": eq_diams,
        }

        for ax, (title, key) in zip(axes, titles):
            values = data_map[key]
            ax.hist(values, bins=self.bin_count, color="steelblue",
                    edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(values), color="red", linestyle="--",
                       linewidth=1.5, label=f"均值: {np.mean(values):.2f} mm")
            ax.axvline(np.median(values), color="green", linestyle="--",
                       linewidth=1.5, label=f"中位数: {np.median(values):.2f} mm")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("粒径 (mm)")
            ax.set_ylabel("颗粒数量")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"{name}_histograms.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("直方图已保存: %s", path)

    def _plot_cumulative_distribution(
        self, measurements: List[ParticleMeasurement], name: str
    ):
        """绘制粒径累计分布曲线。"""
        long_axes = sorted([m.long_axis_mm for m in measurements])
        short_axes = sorted([m.short_axis_mm for m in measurements])
        eq_diams = sorted([m.equivalent_diameter_mm for m in measurements])

        n = len(long_axes)
        cumulative = np.arange(1, n + 1) / n * 100

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(long_axes, cumulative, "b-", linewidth=2, label="长轴")
        ax.plot(short_axes, cumulative, "g-", linewidth=2, label="短轴")
        ax.plot(eq_diams, cumulative, "r--", linewidth=2, label="等效直径")

        for pct, ls, label in [(10, "--", "D10"), (50, "-.", "D50"), (90, ":", "D90")]:
            ax.axhline(pct, color="gray", linestyle=ls, alpha=0.5)
            ax.annotate(
                label,
                xy=(ax.get_xlim()[1] * 0.02, pct),
                fontsize=9,
                color="gray",
            )

        ax.set_title("颗粒粒径累计分布曲线", fontsize=14, fontweight="bold")
        ax.set_xlabel("粒径 (mm)")
        ax.set_ylabel("累计百分比 (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"{name}_cumulative.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("累计分布图已保存: %s", path)

    def _plot_summary_dashboard(
        self, measurements: List[ParticleMeasurement], name: str
    ):
        """绘制综合信息看板。"""
        long_axes = [m.long_axis_mm for m in measurements]
        short_axes = [m.short_axis_mm for m in measurements]
        eq_diams = [m.equivalent_diameter_mm for m in measurements]
        axis_ratios = [m.axis_ratio for m in measurements]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax1 = axes[0, 0]
        ax1.scatter(long_axes, short_axes, c=eq_diams, cmap="viridis",
                    alpha=0.6, edgecolors="black", linewidth=0.3)
        ax1.plot([min(long_axes), max(long_axes)],
                 [min(long_axes), max(long_axes)], "r--", alpha=0.5)
        ax1.set_xlabel("长轴 (mm)")
        ax1.set_ylabel("短轴 (mm)")
        ax1.set_title("长轴 vs 短轴 (颜色=等效直径)", fontweight="bold")
        cbar1 = fig.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label("等效直径 (mm)")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.boxplot(eq_diams, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7))
        ax2.set_ylabel("等效直径 (mm)")
        ax2.set_title("等效直径分布 (箱线图)", fontweight="bold")
        ax2.set_xticklabels(["等效直径"])
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = axes[1, 0]
        ax3.hist(axis_ratios, bins=self.bin_count, color="coral",
                 edgecolor="white", alpha=0.85)
        ax3.set_xlabel("长宽比")
        ax3.set_ylabel("颗粒数量")
        ax3.set_title("颗粒长宽比分布", fontweight="bold")
        ax3.axvline(np.mean(axis_ratios), color="red", linestyle="--",
                    label=f"均值: {np.mean(axis_ratios):.2f}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if measurements:
            sizes = [m.equivalent_diameter_mm for m in measurements]
            categories = []
            bounds = [0, 20, 50, 100, 200, 500, float("inf")]
            labels = ["<20", "20-50", "50-100", "100-200", "200-500", ">500"]
            for s in sizes:
                for lb, ub, lbl in zip(bounds[:-1], bounds[1:], labels):
                    if lb <= s < ub:
                        categories.append(lbl)
                        break
                else:
                    categories.append(labels[-1])

            counts = pd.Series(categories).value_counts()
            counts = counts.reindex(labels).fillna(0)
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(labels)))
            ax4.bar(counts.index, counts.values, color=colors, edgecolor="white")
            ax4.set_xlabel("粒径范围 (mm)")
            ax4.set_ylabel("颗粒数量")
            ax4.set_title("粒径分级统计", fontweight="bold")
            ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = self.output_dir / f"{name}_dashboard.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("综合看板已保存: %s", path)

    def _export_csv(
        self,
        measurements: List[ParticleMeasurement],
        report: StatisticalReport,
        name: str,
    ):
        """导出测量结果和统计报告为 CSV。"""
        records = []
        for m in measurements:
            records.append({
                "particle_id": m.particle_id,
                "x1": m.bbox[0], "y1": m.bbox[1],
                "x2": m.bbox[2], "y2": m.bbox[3],
                "area_mm2": round(m.area_mm2, 4),
                "long_axis_mm": round(m.long_axis_mm, 4),
                "short_axis_mm": round(m.short_axis_mm, 4),
                "equivalent_diameter_mm": round(m.equivalent_diameter_mm, 4),
                "axis_ratio": round(m.axis_ratio, 4),
                "centroid_x": round(m.centroid[0], 2),
                "centroid_y": round(m.centroid[1], 2),
                "orientation_deg": round(m.orientation_deg, 2),
            })

        df = pd.DataFrame(records)
        path = self.output_dir / f"{name}_particles.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("颗粒数据已导出: %s (%d 条记录)", path, len(records))

        stats_records = []
        for metric_name, metric_data in [
            ("长轴 (mm)", "long_axis_mm"),
            ("短轴 (mm)", "short_axis_mm"),
            ("等效直径 (mm)", "equivalent_diameter_mm"),
            ("面积 (mm²)", "area_mm2"),
        ]:
            stats = getattr(report, metric_data, {})
            row = {"指标": metric_name}
            row.update(stats)
            stats_records.append(row)

        stats_df = pd.DataFrame(stats_records)
        stats_path = self.output_dir / f"{name}_statistics.csv"
        stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
        logger.info("统计报告已导出: %s", stats_path)

    def _save_report_txt(self, report: StatisticalReport, name: str):
        """保存可读的文本统计报告。"""
        lines = []
        lines.append("=" * 60)
        lines.append("  堰塞坝表层颗粒粒径检测统计报告")
        lines.append("=" * 60)
        lines.append(f"\n检测颗粒总数: {report.total_particles}\n")

        for label, stats_key in [
            ("长轴 (mm)", "long_axis_mm"),
            ("短轴 (mm)", "short_axis_mm"),
            ("等效直径 (mm)", "equivalent_diameter_mm"),
            ("面积 (mm²)", "area_mm2"),
        ]:
            stats = getattr(report, stats_key, {})
            if not stats:
                continue
            lines.append(f"--- {label} ---")
            lines.append(f"  均值 (Mean):     {stats.get('mean', 'N/A'):.2f}")
            lines.append(f"  标准差 (Std):    {stats.get('std', 'N/A'):.2f}")
            lines.append(f"  最小值 (Min):    {stats.get('min', 'N/A'):.2f}")
            lines.append(f"  D10:             {stats.get('d10', 'N/A'):.2f}")
            lines.append(f"  D25:             {stats.get('d25', 'N/A'):.2f}")
            lines.append(f"  D50 (中位数):    {stats.get('d50', 'N/A'):.2f}")
            lines.append(f"  D75:             {stats.get('d75', 'N/A'):.2f}")
            lines.append(f"  D90:             {stats.get('d90', 'N/A'):.2f}")
            lines.append(f"  最大值 (Max):    {stats.get('max', 'N/A'):.2f}")
            lines.append("")

        lines.append("=" * 60)

        text = "\n".join(lines)
        path = self.output_dir / f"{name}_report.txt"
        path.write_text(text, encoding="utf-8")
        logger.info("文本报告已保存: %s", path)

    def _save_overlay_image(
        self,
        image: np.ndarray,
        measurements: List[ParticleMeasurement],
        name: str,
    ):
        """保存检测和分割结果叠加图。"""
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        np.random.seed(42)
        for m in measurements:
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))

            if m.mask is not None:
                mask_overlay = np.zeros_like(overlay)
                mask_overlay[m.mask > 0] = color
                overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)

            x1, y1, x2, y2 = [int(v) for v in m.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"#{m.particle_id}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        path = self.output_dir / f"{name}_overlay.png"
        cv2.imwrite(str(path), overlay)
        logger.info("叠加图已保存: %s", path)

    # ============================================================
    # 论文评估可视化 — 符合 Minerals 2024 论文图表格式
    # ============================================================

    def plot_particle_size_distribution(
        self,
        measurements: List[ParticleMeasurement],
        name: str,
    ):
        """
        绘制粒径分布直方图 + Rosin-Rammler 概率密度曲线 — 论文 Figure 6/10 格式。

        直方图展示实测数据，叠加红色 RR 拟合概率密度曲线。

        Args:
            measurements: 颗粒测量结果列表
            name:        输出文件名前缀
        """
        from metrics import fit_rosin_rammler, rosin_rammler_pdf

        diameters = np.array([m.equivalent_diameter_mm for m in measurements])
        if len(diameters) == 0:
            return

        Xm, n, rr_r2 = fit_rosin_rammler(diameters)

        fig, ax = plt.subplots(figsize=(10, 6))

        counts, bins, _ = ax.hist(
            diameters, bins=self.bin_count,
            color="steelblue", edgecolor="white",
            alpha=0.75, density=True,
            label=f"实测分布 (n={len(diameters)})"
        )

        x_smooth = np.linspace(bins[0], bins[-1], 200)
        pdf_values = rosin_rammler_pdf(x_smooth, Xm, n)
        ax.plot(
            x_smooth, pdf_values, "r-", linewidth=2.5,
            label=f"RR 拟合 (Xm={Xm:.1f} mm, n={n:.2f}, R\u00b2={rr_r2:.3f})"
        )

        ax.set_xlabel("等效粒径 (mm)", fontsize=12)
        ax.set_ylabel("概率密度", fontsize=12)
        ax.set_title("颗粒粒径分布 (Particle Size Distribution)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"{name}_size_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("粒径分布图已保存: %s", path)

    def plot_cumulative_passing_with_rr(
        self,
        measurements: List[ParticleMeasurement],
        name: str,
    ):
        """
        绘制累积通过百分比曲线 + Rosin-Rammler 拟合 + 特征粒径标注
        — 论文 Figure 7/11/14 格式。

        显示实测累积曲线（蓝色）和 RR 拟合曲线（红色虚线），
        并在 X10~X100 位置添加垂直参考线。

        Args:
            measurements: 颗粒测量结果列表
            name:        输出文件名前缀
        """
        from metrics import (
            fit_rosin_rammler, rosin_rammler_cdf,
            compute_characteristic_sizes, calc_rr_characteristic_sizes,
        )

        diameters = np.array([m.equivalent_diameter_mm for m in measurements])
        if len(diameters) == 0:
            return

        sorted_d = np.sort(diameters)
        n_total = len(sorted_d)
        cumulative = np.arange(1, n_total + 1) / n_total * 100

        Xm, n, rr_r2 = fit_rosin_rammler(diameters)
        char_sizes = compute_characteristic_sizes(diameters)

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(sorted_d, cumulative, "b-", linewidth=2.5, label=f"实测数据 (n={n_total})")

        if Xm > 0 and n > 0:
            x_fit = np.linspace(sorted_d[0], sorted_d[-1], 300)
            y_fit = rosin_rammler_cdf(x_fit, Xm, n) * 100
            ax.plot(
                x_fit, y_fit, "r--", linewidth=2,
                label=f"RR 拟合 (Xm={Xm:.1f} mm, n={n:.2f}, R\u00b2={rr_r2:.3f})"
            )

        colors_10 = plt.cm.tab10(np.linspace(0, 1, 10))
        for idx, (pct, size) in enumerate(sorted(char_sizes.items())):
            ax.axvline(size, color=colors_10[idx], linestyle=":", alpha=0.7, linewidth=1)
            ax.annotate(
                f"X{pct}={size:.1f}",
                xy=(size, pct),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=colors_10[idx],
                rotation=45,
            )

        ax.set_xlabel("等效粒径 (mm)", fontsize=12)
        ax.set_ylabel("累积通过百分比 (%)", fontsize=12)
        ax.set_title("累积粒径分布曲线 (Cumulative Passing Percentage)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        path = self.output_dir / f"{name}_cumulative_rr.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("累积分布+RR拟合图已保存: %s", path)

    def plot_characteristic_sizes_comparison(
        self,
        pred_char_sizes: Dict[int, float],
        gt_char_sizes: Dict[int, float],
        name: str,
    ):
        """
        绘制特征粒径对比柱状图 — 论文 Table 4 可视化。

        将预测和真值的 X10~X100 特征粒径以分组柱状图对比显示。

        Args:
            pred_char_sizes: 预测的特征粒径 {10: X10, 20: X20, ...}
            gt_char_sizes:   真值的特征粒径
            name:            输出文件名前缀
        """
        percentiles = sorted(pred_char_sizes.keys())
        pred_vals = [pred_char_sizes.get(p, 0) for p in percentiles]
        gt_vals = [gt_char_sizes.get(p, 0) for p in percentiles]

        x = np.arange(len(percentiles))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        bars1 = ax.bar(x - width / 2, gt_vals, width, label="真值 (Ground Truth)",
                       color="steelblue", edgecolor="white")
        bars2 = ax.bar(x + width / 2, pred_vals, width, label="预测值 (Predicted)",
                       color="coral", edgecolor="white")

        ax.set_xlabel("特征粒径", fontsize=12)
        ax.set_ylabel("粒径 (mm)", fontsize=12)
        ax.set_title("特征粒径对比 (X10 ~ X100)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"X{p}" for p in percentiles])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars2, pred_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

        plt.tight_layout()
        path = self.output_dir / f"{name}_char_sizes_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("特征粒径对比图已保存: %s", path)

    def plot_evaluation_metrics_bars(
        self,
        metrics_dict: Dict[str, float],
        name: str,
        title: str = "评估指标对比",
    ):
        """
        绘制评估指标对比柱状图。

        用于展示 PA / mIOU / Dice / R² 等多个指标的对比。

        Args:
            metrics_dict: {"指标名": 数值, ...}
            name:         输出文件名前缀
            title:        图表标题
        """
        labels = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_ylabel("指标值", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = self.output_dir / f"{name}_metrics_bars.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("指标对比柱状图已保存: %s", path)

    def save_evaluation_report(
        self,
        report_text: str,
        name: str,
    ):
        """
        保存评估报告文本文件。

        Args:
            report_text: 报告文本内容
            name:        输出文件名前缀
        """
        path = self.output_dir / f"{name}_evaluation_report.txt"
        path.write_text(report_text, encoding="utf-8")
        logger.info("评估报告已保存: %s", path)