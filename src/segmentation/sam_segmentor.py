import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

logger = logging.getLogger(__name__)


class SAMSegmentor:
    """基于 Segment Anything Model (SAM) 的颗粒分割器。

    使用 YOLOv8 检测框作为 Prompt，驱动 SAM 生成高精度的
    颗粒实例分割掩码，实现像素级颗粒边界提取。
    """

    SUPPORTED_MODELS = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            device = "cpu"

        self.model_type = model_type
        self.device = device

        if checkpoint_path is None:
            default_name = self.SUPPORTED_MODELS.get(model_type)
            if default_name is None:
                raise ValueError(
                    f"不支持的模型类型: {model_type}，"
                    f"可选: {list(self.SUPPORTED_MODELS.keys())}"
                )
            checkpoint_path = default_name

        logger.info("加载 SAM 模型: %s", checkpoint_path)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        logger.info("SAM 分割器初始化完成, 设备: %s", device)

    def set_image(self, image: np.ndarray):
        """设置待分割的图像。

        Args:
            image: RGB 格式的 numpy 数组, shape (H, W, 3)。
        """
        self.predictor.set_image(image)

    def segment_with_boxes(
        self,
        bboxes: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[dict]:
        """使用边界框作为 Prompt 生成分割掩码。

        Args:
            bboxes: 边界框数组, shape (N, 4), 格式 (x1, y1, x2, y2)。
            image_shape: 图像尺寸 (H, W)，用于修正超出边界的框。

        Returns:
            分割结果列表，每项包含:
                - mask: 二值分割掩码 (H, W)
                - bbox: 对应的边界框
                - area: 掩码面积 (像素)
                - score: SAM 的分割质量评分
        """
        if len(bboxes) == 0:
            return []

        if image_shape is not None:
            bboxes = self._clip_bboxes(bboxes, image_shape)

        input_boxes = torch.tensor(bboxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            input_boxes, self.predictor.original_size
        )

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

        results = []
        for i in range(len(bboxes)):
            mask = masks[i, 0]
            results.append({
                "mask": mask,
                "bbox": tuple(bboxes[i].tolist()),
                "area": int(mask.sum()),
                "score": float(scores[i]),
            })

        logger.info("SAM 生成 %d 个分割掩码", len(results))
        return results

    def refine_masks(
        self,
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """对掩码进行后处理：去除噪声、填充孔洞、平滑边界。"""
        refined = []
        for mask in masks:
            refined_mask = self._postprocess_mask(mask.astype(np.uint8))
            refined.append(refined_mask)
        return refined

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """单掩码后处理流水线。"""
        from scipy import ndimage

        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        from skimage.morphology import remove_small_objects

        mask = remove_small_objects(mask.astype(bool), min_size=50)
        mask = mask.astype(np.uint8)

        structure = np.ones((3, 3))
        mask = ndimage.binary_opening(mask, structure=structure).astype(np.uint8)
        mask = ndimage.binary_closing(mask, structure=structure).astype(np.uint8)

        return mask

    @staticmethod
    def _clip_bboxes(
        bboxes: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """将边界框裁剪到图像范围内。"""
        h, w = image_shape
        clipped = bboxes.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
        clipped[:, 2] = np.clip(clipped[:, 2], 0, w - 1)
        clipped[:, 3] = np.clip(clipped[:, 3], 0, h - 1)
        return clipped