import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path as CogPath

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


class Predictor(BasePredictor):
    def setup(self):
        """Load the LivePortrait model and weights"""
        inference_cfg = InferenceConfig()
        crop_cfg = CropConfig()

        inference_cfg.flag_force_cpu = False
        inference_cfg.device_id = 0
        inference_cfg.flag_use_half_precision = True

        self.pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )

        print("LivePortrait model loaded successfully")

    def predict(
        self,
        source_image: CogPath = Input(description="Source portrait image (face photo)"),
        driving_video: CogPath = Input(description="Driving video with expressions to mimic"),
    ) -> CogPath:
        """Animate the source portrait using expressions from the driving video"""
        output_dir = tempfile.mkdtemp()

        args = ArgumentConfig()
        args.source = str(source_image)
        args.driving = str(driving_video)
        args.output_dir = output_dir

        if not os.path.exists(args.source):
            raise FileNotFoundError(f"Source image not found: {args.source}")
        if not os.path.exists(args.driving):
            raise FileNotFoundError(f"Driving video not found: {args.driving}")

        print(f"Processing source: {args.source}")
        print(f"Processing driving: {args.driving}")
        wfp, wfp_concat = self.pipeline.execute(args)

        return CogPath(Path(wfp))
