import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.detection.yolo_detector import YOLODetector
from src.measurement.particle_measurer import ParticleMeasurer
from src.segmentation.sam_segmentor import SAMSegmentor
from src.statistics.analyzer import ParticleAnalyzer, StatisticalReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DamParticlePipeline")


class DamParticlePipeline:
    """堰塞坝表层颗粒物质智能检测分析流水线。

    整合 YOLOv8 检测 → SAM 分割 → 粒径测量 → 统计分析
    的完整处理链条。

    使用方式:
        pipeline = DamParticlePipeline("config/config.yaml")
        pipeline.run("path/to/uav_image.jpg")
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.yolo_detector = YOLODetector(
            model_path=self.config["yolo"]["model_path"],
            conf_threshold=self.config["yolo"]["conf_threshold"],
            iou_threshold=self.config["yolo"]["iou_threshold"],
            device=self.config["yolo"]["device"],
            target_classes=self.config["yolo"].get("classes"),
        )

        self.sam_segmentor = SAMSegmentor(
            model_type=self.config["sam"]["model_type"],
            checkpoint_path=self.config["sam"]["checkpoint_path"],
            device=self.config["sam"]["device"],
        )

        meas_cfg = self.config["measurement"]
        self.measurer = ParticleMeasurer(
            pixel_size_mm=meas_cfg.get("pixel_size_mm"),
            reference_length_pixels=meas_cfg.get("reference_length_pixels"),
            reference_length_mm=meas_cfg.get("reference_length_mm"),
            min_area_pixels=meas_cfg["min_area_pixels"],
            max_area_pixels=meas_cfg["max_area_pixels"],
            long_short_axis_ratio_max=meas_cfg["long_short_axis_ratio_max"],
        )

        stat_cfg = self.config["statistics"]
        self.analyzer = ParticleAnalyzer(
            output_dir=stat_cfg.get("output_dir", "outputs"),
            bin_count=stat_cfg["bin_count"],
            save_masks=stat_cfg.get("save_masks", True),
            save_overlay=stat_cfg.get("save_overlay", True),
        )

        self.output_dir = Path(self.config["data"].get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("流水线初始化完成")

    def run(
        self,
        image_path: str,
        image_name: Optional[str] = None,
        scale_mm_per_pixel: Optional[float] = None,
    ) -> dict:
        """执行完整的颗粒检测分析流水线。

        Args:
            image_path: 输入影像路径 (RGB, JPG/PNG/TIF)。
            image_name: 输出文件前缀 (默认使用输入文件名)。
            scale_mm_per_pixel: 像素分辨率 (mm/pixel)，如未指定则使用配置文件中的设置。

        Returns:
            包含检测、分割、测量、统计结果的字典。
        """
        t_start = time.time()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"输入影像不存在: {image_path}")

        if image_name is None:
            image_name = image_path.stem

        logger.info("=" * 60)
        logger.info("开始处理: %s", image_path.name)
        logger.info("=" * 60)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取影像: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w = image_rgb.shape[:2]
        logger.info("影像尺寸: %d x %d", image_w, image_h)

        if scale_mm_per_pixel is not None:
            self.measurer._scale_mm_per_pixel = scale_mm_per_pixel
            logger.info("比例尺: %.4f mm/pixel", scale_mm_per_pixel)

        t1 = time.time()
        detections = self.yolo_detector.detect(image_rgb)
        t2 = time.time()
        logger.info("YOLOv8 检测耗时: %.1fs, 检测到 %d 个目标", t2 - t1, len(detections))

        if len(detections) == 0:
            logger.warning("未检测到任何颗粒，流水线终止")
            return {
                "detections": [],
                "segmentations": [],
                "measurements": [],
                "report": None,
                "total_time": time.time() - t_start,
            }

        bboxes = self.yolo_detector.get_bboxes_array(detections)

        t3 = time.time()
        self.sam_segmentor.set_image(image_rgb)
        segmentations = self.sam_segmentor.segment_with_boxes(
            bboxes, image_shape=(image_h, image_w)
        )
        t4 = time.time()
        logger.info("SAM 分割耗时: %.1fs, 生成 %d 个掩码", t4 - t3, len(segmentations))

        t5 = time.time()
        measurements = self.measurer.measure(segmentations, detections)
        t6 = time.time()
        logger.info("粒径测量耗时: %.1fs, 有效颗粒 %d 个", t6 - t5, len(measurements))

        t7 = time.time()
        report = self.analyzer.analyze(measurements, image, image_name)
        t8 = time.time()
        logger.info("统计分析耗时: %.1fs", t8 - t7)

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info("处理完成! 总耗时: %.1fs", total_time)
        logger.info("检测颗粒: %d → 有效颗粒: %d", len(detections), len(measurements))
        logger.info("=" * 60)

        return {
            "detections": detections,
            "segmentations": segmentations,
            "measurements": measurements,
            "report": report,
            "total_time": total_time,
        }

    def run_batch(
        self,
        image_dir: str,
        pattern: str = "*.jpg",
        scale_mm_per_pixel: Optional[float] = None,
    ) -> dict:
        """批量处理影像目录。

        Args:
            image_dir: 影像目录路径。
            pattern: 文件匹配模式。
            scale_mm_per_pixel: 统一的像素分辨率。

        Returns:
            所有影像的处理结果字典。
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob(pattern))
        image_files.extend(sorted(image_dir.glob("*.png")))
        image_files.extend(sorted(image_dir.glob("*.tif")))
        image_files.extend(sorted(image_dir.glob("*.tiff")))

        image_files = sorted(set(image_files))

        if not image_files:
            raise FileNotFoundError(f"目录 {image_dir} 中未找到匹配的影像文件")

        logger.info("批量处理 %d 张影像", len(image_files))

        all_results = {}
        for img_path in image_files:
            try:
                result = self.run(
                    str(img_path),
                    image_name=img_path.stem,
                    scale_mm_per_pixel=scale_mm_per_pixel,
                )
                all_results[str(img_path)] = result
            except Exception as e:
                logger.error("处理 %s 失败: %s", img_path.name, e)
                all_results[str(img_path)] = {"error": str(e)}

        total_particles = sum(
            len(r.get("measurements", [])) for r in all_results.values()
            if "error" not in r
        )
        logger.info("批量处理完成: %d 张影像, 总计 %d 个颗粒",
                    len(image_files), total_particles)

        return all_results