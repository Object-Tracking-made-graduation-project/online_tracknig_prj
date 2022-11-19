from dataclasses import dataclass, field
from marshmallow import Schema, fields
import torch
import typing


@dataclass()
class ModelParams:
    aspect_ratio_thresh: float
    camid: int
    ckpt: str
    conf: float = field(metadata={"allow_none": True})
    demo: str
    device: str
    exp_file: str
    experiment_name: str
    fp16: bool
    fps: int
    fuse: bool
    match_thresh: float
    min_box_area: int
    mot20: bool
    name: str = field(metadata={"allow_none": True})
    nms: float = field(metadata={"allow_none": True})
    path: str
    save_result: bool
    track_buffer: int
    track_thresh: float
    trt: bool
    tsize: int = field(metadata={"allow_none": True})


@dataclass()
class InferenceParams:
    model_params: ModelParams
