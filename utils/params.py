from dataclasses import dataclass, field
from marshmallow import Schema, fields
import torch
import typing

URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"


@dataclass()
class ModelParams:
    pass


@dataclass()
class ServiceParams:
    model_params: ModelParams = None
    use_model: str = "iim"
    video_url: str = field(default=URL_MODEL)
    stream: int = 0
