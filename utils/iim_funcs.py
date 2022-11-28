import logging

import cv2
import numpy as np

from detection_models.iim.misc.image import get_points_on_image
from detection_models.iim.model.utils import load_model
from utils.params import ModelParams
from utils.params import BaseModel


class IimModel(BaseModel):
    def __init__(self, model_params: ModelParams):
        self.config = model_params
        self.model = load_model(model_params)

    def online_inference(self, frame: np.ndarray) -> np.ndarray:
        points = get_points_on_image(frame, self.model)
        count = len(points)

        size = int(np.floor(10 - np.log10(1 + count)))
        if size < 1:
            size = 1
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, str(count), (100, 100), font, 4, (0, 255, 255), 10, cv2.LINE_AA)

        logging.info("Found %d persons", count)
        print(f"Found {count} persons", count)

        return frame
