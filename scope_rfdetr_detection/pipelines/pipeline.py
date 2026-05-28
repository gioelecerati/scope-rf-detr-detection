"""RF-DETR object detection pipeline.

Runs Roboflow's RF-DETR per frame and returns the frame annotated with
bounding boxes for the detected COCO classes.

    detect (RF-DETR) -> annotate (supervision)
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import RFDetrConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class RFDetrDetectionPipeline(Pipeline):
    """Real-time object detection with RF-DETR."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return RFDetrConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        import rfdetr
        import supervision as sv

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        variant: str = kwargs.get("model_variant", "nano")
        model_cls_name = _VARIANTS.get(variant, "RFDETRNano")
        model_cls = getattr(rfdetr, model_cls_name)

        logger.info("Loading RF-DETR model: %s (device=%s)", model_cls_name, self.device)
        self.model = model_cls()

        # COCO class id -> name, used for labels (best-effort).
        self._class_names: dict[int, str] = {}
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES

            self._class_names = dict(COCO_CLASSES)
        except Exception:  # pragma: no cover - depends on rfdetr internals
            logger.warning("Could not import RF-DETR COCO class names; using ids")

        self._box_annotator = sv.BoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()

        logger.info("RF-DETR detection pipeline ready")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for RFDetrDetectionPipeline")

        confidence: float = kwargs.get("confidence_threshold", 0.5)
        show_labels: bool = kwargs.get("show_labels", True)
        show_confidence: bool = kwargs.get("show_confidence", True)

        output_frames: list[torch.Tensor] = []

        for frame_tensor in video:
            frame_np = frame_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(frame_np, mode="RGB")

            detections = self.model.predict(pil_image, threshold=confidence)

            annotated = frame_np.copy()
            annotated = self._box_annotator.annotate(
                scene=annotated, detections=detections
            )
            if show_labels and len(detections) > 0:
                labels = []
                for i in range(len(detections)):
                    class_id = int(detections.class_id[i])
                    name = self._class_names.get(class_id, str(class_id))
                    if show_confidence and detections.confidence is not None:
                        labels.append(f"{name} {detections.confidence[i]:.2f}")
                    else:
                        labels.append(name)
                annotated = self._label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )

            frame_out = torch.from_numpy(annotated.astype(np.float32) / 255.0)
            output_frames.append(frame_out)

        return {"video": torch.stack(output_frames, dim=0)}
