"""Configuration schema for the RF-DETR detection pipeline."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class RFDetrConfig(BasePipelineConfig):
    """Configuration for the RF-DETR object detection pipeline.

    Real-time object detection with Roboflow's RF-DETR. Apache-2.0
    variants (Nano/Small/Medium/Base/Large) detect the 80 COCO classes
    and the frame is returned with annotated bounding boxes.
    """

    pipeline_id: ClassVar[str] = "rfdetr-detection"
    pipeline_name: ClassVar[str] = "RF-DETR Detection"
    pipeline_description: ClassVar[str] = (
        "Real-time object detection using Roboflow RF-DETR. "
        "Detects the 80 COCO classes and overlays annotated bounding boxes."
    )
    supports_prompts: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    # ── Load-time parameters ──────────────────────────────────────

    model_variant: Literal[
        "nano", "small", "medium", "base", "large"
    ] = Field(
        default="nano",
        description=(
            "RF-DETR model size. Nano is fastest (real-time on modest GPUs), "
            "Large is most accurate. All listed variants are Apache-2.0."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Model",
            is_load_param=True,
        ),
    )

    # ── Detection ─────────────────────────────────────────────────

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Minimum confidence score for detections.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Confidence Threshold",
        ),
    )

    show_labels: bool = Field(
        default=True,
        description="Show class labels on bounding boxes.",
        json_schema_extra=ui_field_config(
            order=11,
            label="Show Labels",
        ),
    )

    show_confidence: bool = Field(
        default=True,
        description="Show confidence scores next to labels.",
        json_schema_extra=ui_field_config(
            order=12,
            label="Show Confidence",
        ),
    )
