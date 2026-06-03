import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


@dataclass
class ParticleMeasurement:
    """单个颗粒的粒径测量结果。

    包含论文2024 Minerals Table 1/2/3/4 所需的全部参数。
    """

    particle_id: int
    bbox: tuple
    area_pixels: int
    area_mm2: float
    long_axis_pixels: float
    short_axis_pixels: float
    long_axis_mm: float
    short_axis_mm: float
    equivalent_diameter_pixels: float
    equivalent_diameter_mm: float
    axis_ratio: float
    centroid: tuple
    orientation_deg: float
    perimeter_pixels: float
    perimeter_mm: float = 0.0
    eq_circle_diameter_mm: float = 0.0
    circularity: float = 0.0
    mask: Optional[np.ndarray] = field(repr=False, default=None)


class ParticleMeasurer:
    """颗粒粒径测量器。

    基于分割掩码计算每个颗粒的几何参数：
    - 长轴 (major axis)：最佳拟合椭圆的长轴，通过掩码轮廓的 PCA 计算
    - 短轴 (minor axis)：最佳拟合椭圆的短轴
    - 等效直径 (equivalent diameter)：与颗粒面积相等的圆的直径

    支持两种尺度标定方式：
    1. 直接指定像素分辨率 (pixel_size_mm)
    2. 指定参考线段 (reference_length_pixels + reference_length_mm)
    """

    def __init__(
        self,
        pixel_size_mm: Optional[float] = None,
        reference_length_pixels: Optional[float] = None,
        reference_length_mm: Optional[float] = None,
        min_area_pixels: int = 100,
        max_area_pixels: int = 100000,
        long_short_axis_ratio_max: float = 10.0,
    ):
        self.pixel_size_mm = pixel_size_mm
        self.reference_length_pixels = reference_length_pixels
        self.reference_length_mm = reference_length_mm

        if pixel_size_mm is not None:
            self._scale_mm_per_pixel = pixel_size_mm
        elif reference_length_pixels is not None and reference_length_mm is not None:
            self._scale_mm_per_pixel = reference_length_mm / reference_length_pixels
        else:
            self._scale_mm_per_pixel = None
            logger.warning("未设置像素-毫米比例尺，粒径将以像素为单位输出")

        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.long_short_axis_ratio_max = long_short_axis_ratio_max

    def measure(
        self,
        segmentation_results: List[dict],
        detect_results: Optional[List[dict]] = None,
    ) -> List[ParticleMeasurement]:
        """对所有分割结果执行粒径测量。

        Args:
            segmentation_results: SAM 分割结果列表。
            detect_results: YOLOv8 检测结果列表 (可选，用于补充信息)。

        Returns:
            ParticleMeasurement 列表。
        """
        measurements = []
        valid_idx = 0

        for i, seg in enumerate(segmentation_results):
            mask = seg["mask"]

            if mask.sum() < self.min_area_pixels:
                continue
            if mask.sum() > self.max_area_pixels:
                continue

            geo = self._compute_geometry(mask)

            long_axis_px = geo["long_axis"]
            short_axis_px = geo["short_axis"]

            if short_axis_px > 0:
                axis_ratio = long_axis_px / short_axis_px
            else:
                axis_ratio = float("inf")

            if axis_ratio > self.long_short_axis_ratio_max:
                logger.debug(
                    "颗粒 %d 长宽比 %.2f 超过阈值 %.2f，跳过",
                    i, axis_ratio, self.long_short_axis_ratio_max,
                )
                continue

            scale = self._scale_mm_per_pixel or 1.0

            from metrics import equivalent_circle_diameter as _ecd, circularity as _circ

            area_mm2_val = float(mask.sum()) * scale ** 2 if self._scale_mm_per_pixel else -1.0
            perim_px = geo["perimeter"]
            perim_mm_val = perim_px * scale
            eq_cd_mm = float(_ecd(np.array([area_mm2_val]))[0]) if area_mm2_val > 0 else 0.0
            circ_val = float(_circ(np.array([area_mm2_val]), np.array([perim_mm_val]))[0]) if perim_mm_val > 0 else 0.0

            measurement = ParticleMeasurement(
                particle_id=valid_idx,
                bbox=seg.get("bbox", (0, 0, 0, 0)),
                area_pixels=int(mask.sum()),
                area_mm2=area_mm2_val,
                long_axis_pixels=long_axis_px,
                short_axis_pixels=short_axis_px,
                long_axis_mm=long_axis_px * scale,
                short_axis_mm=short_axis_px * scale,
                equivalent_diameter_pixels=geo["equivalent_diameter"],
                equivalent_diameter_mm=geo["equivalent_diameter"] * scale,
                axis_ratio=axis_ratio,
                centroid=geo["centroid"],
                orientation_deg=geo["orientation_deg"],
                perimeter_pixels=perim_px,
                perimeter_mm=perim_mm_val,
                eq_circle_diameter_mm=eq_cd_mm,
                circularity=circ_val,
                mask=mask,
            )
            measurements.append(measurement)
            valid_idx += 1

        logger.info(
            "完成 %d 个颗粒测量 (过滤 %d 个)",
            len(measurements),
            len(segmentation_results) - len(measurements),
        )
        return measurements

    def _compute_geometry(self, mask: np.ndarray) -> dict:
        """计算单个掩码的几何参数。

        使用掩码轮廓点的 PCA 计算长轴和短轴。
        """
        binary_mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return self._fallback_geometry(binary_mask)

        all_points = np.vstack(contours).squeeze()

        if all_points.ndim != 2 or len(all_points) < 5:
            return self._fallback_geometry(binary_mask)

        mean = all_points.mean(axis=0)
        centered = all_points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        long_axis = 4.0 * np.sqrt(eigenvalues[0])
        short_axis = 4.0 * np.sqrt(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        orientation_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        orientation_deg = np.degrees(orientation_rad)
        orientation_deg = orientation_deg % 180

        props = regionprops(binary_mask)
        if props:
            eq_diameter = props[0].equivalent_diameter_area
            perimeter = props[0].perimeter
            centroid = (props[0].centroid[1], props[0].centroid[0])
        else:
            eq_diameter = 2.0 * np.sqrt(binary_mask.sum() / np.pi)
            perimeter = 0.0
            y_idx, x_idx = np.where(binary_mask)
            centroid = (float(x_idx.mean()), float(y_idx.mean()))

        return {
            "long_axis": float(long_axis),
            "short_axis": float(short_axis),
            "equivalent_diameter": float(eq_diameter),
            "orientation_deg": float(orientation_deg),
            "perimeter": float(perimeter),
            "centroid": centroid,
        }

    @staticmethod
    def _fallback_geometry(mask: np.ndarray) -> dict:
        """当轮廓提取失败时的回退几何计算。"""
        area = mask.sum()
        eq_diameter = 2.0 * np.sqrt(area / np.pi)

        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            return {
                "long_axis": 0.0, "short_axis": 0.0,
                "equivalent_diameter": 0.0, "orientation_deg": 0.0,
                "perimeter": 0.0, "centroid": (0.0, 0.0),
            }

        centroid = (float(x_idx.mean()), float(y_idx.mean()))
        return {
            "long_axis": eq_diameter,
            "short_axis": eq_diameter,
            "equivalent_diameter": float(eq_diameter),
            "orientation_deg": 0.0,
            "perimeter": float(eq_diameter * np.pi),
            "centroid": centroid,
        }

    @staticmethod
    def set_scale_from_reference(
        ref_point1: Tuple[float, float],
        ref_point2: Tuple[float, float],
        known_distance_mm: float,
    ) -> float:
        """根据图像中已知距离的参考点和实际毫米距离计算比例尺。

        Args:
            ref_point1: 参考点1的图像坐标 (x1, y1)
            ref_point2: 参考点2的图像坐标 (x2, y2)
            known_distance_mm: 两点之间的实际距离，单位毫米

        Returns:
            每个像素对应的毫米数
        """
        x1, y1 = ref_point1
        x2, y2 = ref_point2
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        scale = known_distance_mm / pixel_distance
        logger.info(
            "比例尺: %.4f mm/pixel (%.1f mm / %.1f pixels)",
            scale, known_distance_mm, pixel_distance,
        )
        return scale

    @staticmethod
    def compute_particle_size_distribution(
        measurements: List,
    ) -> dict:
        """
        计算颗粒粒径分布统计数据。

        对所有颗粒的等效直径进行统计分析，输出论文所需的分布参数。

        Args:
            measurements: ParticleMeasurement 列表

        Returns:
            包含以下字段的字典:
                - diameters_mm: 等效直径数组 (mm)
                - areas_mm2: 面积数组 (mm²)
                - perimeters_mm: 周长数组 (mm)
                - circularities: 圆度数组
                - axis_ratios: 长宽比数组
        """
        if len(measurements) == 0:
            return {
                "diameters_mm": np.array([]),
                "areas_mm2": np.array([]),
                "perimeters_mm": np.array([]),
                "circularities": np.array([]),
                "axis_ratios": np.array([]),
            }

        diameters = np.array([m.equivalent_diameter_mm for m in measurements])
        areas = np.array([m.area_mm2 for m in measurements])
        perimeters = np.array([m.perimeter_mm for m in measurements])
        circularities = np.array([m.circularity for m in measurements])
        ratios = np.array([m.axis_ratio for m in measurements])

        return {
            "diameters_mm": diameters,
            "areas_mm2": areas,
            "perimeters_mm": perimeters,
            "circularities": circularities,
            "axis_ratios": ratios,
        }

    @staticmethod
    def fit_rosin_rammler_distribution(
        measurements: List,
    ) -> Tuple[float, float, float]:
        """
        调用 metrics.py 中的函数，对粒径数据拟合 Rosin-Rammler 分布。

        Args:
            measurements: ParticleMeasurement 列表

        Returns:
            (Xm, n, R²) — 特征粒径 X50 (mm), 均匀性指数, 拟合决定系数
        """
        from metrics import fit_rosin_rammler

        if len(measurements) == 0:
            return (0.0, 1.0, 0.0)

        diameters = np.array([m.equivalent_diameter_mm for m in measurements])
        return fit_rosin_rammler(diameters)