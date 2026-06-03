import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """基于 YOLOv8 的堰塞坝颗粒检测器。

    使用预训练或微调的 YOLOv8 模型检测影像中的石块颗粒，
    输出每个颗粒的边界框坐标、置信度和类别信息。
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        target_classes: Optional[List[int]] = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            device = "cpu"

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes

        self.model = YOLO(model_path)
        self.model.to(device)

        logger.info("YOLOv8 检测器初始化完成, 设备: %s", device)

    def detect(self, image: np.ndarray) -> List[dict]:
        """对输入图像执行颗粒检测。

        Args:
            image: BGR 格式的 numpy 数组, shape (H, W, 3)。

        Returns:
            检测结果列表，每项包含:
                - bbox: (x1, y1, x2, y2) 边界框坐标
                - confidence: 置信度分数
                - class_id: 类别 ID
                - class_name: 类别名称
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                class_id = cls_ids[i]
                if self.target_classes is not None and class_id not in self.target_classes:
                    continue

                class_name = self.model.names.get(class_id, str(class_id))
                detections.append({
                    "bbox": tuple(boxes[i].tolist()),
                    "confidence": float(confs[i]),
                    "class_id": class_id,
                    "class_name": class_name,
                })

        logger.info("检测到 %d 个颗粒目标", len(detections))
        return detections

    def get_bbox_centers(self, detections: List[dict]) -> List[Tuple[float, float]]:
        """从检测结果中提取边界框中心坐标。"""
        centers = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return centers

    def get_bboxes_array(self, detections: List[dict]) -> np.ndarray:
        """将检测框转换为 (N, 4) 的 numpy 数组。"""
        return np.array([det["bbox"] for det in detections], dtype=np.float32)