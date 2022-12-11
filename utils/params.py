from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

import numpy as np

URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"


@dataclass()
class ModelParams:
    pass


# TODO: make it abstract
class BaseModel:
    def __init__(self, model_params: ModelParams):
        pass

    def online_inference(self, frame: np.ndarray, video_mask: np.ndarray = None) -> np.ndarray:
        """
        функция для инференса
        """
        pass


class Mode(IntEnum):
    ORIGINAL = 1
    IIM = 2
    BYTETRACK = 3


@dataclass()
class ServiceParams:
    use_models: str = field(default_factory=lambda: "iim,bytetrack")
    frames_num_before_show: int = 2
    stream: int = 0
    interval: Dict[int, float] = field(
        default_factory=lambda: {Mode.ORIGINAL: 0.100, Mode.IIM: 0.02, Mode.BYTETRACK: 0.02})
