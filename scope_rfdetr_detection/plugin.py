"""Scope plugin hook implementation for RF-DETR detection."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pipeline import RFDetrDetectionPipeline

    register(RFDetrDetectionPipeline)
