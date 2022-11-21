import cv2
import numpy as np

from detection_models.iim.misc.image import get_points_on_image
from detection_models.iim.model.utils import load_model
from utils.funcs import BaseModel
from utils.params import ModelParams


class IimModel(BaseModel):
    def __init__(self, model_params: ModelParams):
        self.config = model_params
        self.model = load_model(model_params)

    def online_inference(self, frame: np.ndarray) -> np.ndarray:
        points = get_points_on_image(frame, self.model)
        size = 5
        for x, y in points:
            frame = cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)

        return frame