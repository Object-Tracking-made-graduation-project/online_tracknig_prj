import logging

import cv2
import numpy as np

from detection_models.iim.misc.image import get_points_on_image
from detection_models.iim.misc.params import IimParams
from detection_models.iim.model.utils import load_model
from utils.params import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IimModel(BaseModel):
    def __init__(self, model_params: IimParams):
        self.config = model_params
        self.model = load_model(model_params)
        self.history = np.full(self.config.smoothing_queue_size, np.nan)
        self.counter = 0

    def online_inference(self, frame: np.ndarray) -> np.ndarray:
        points = get_points_on_image(frame, self.model)
        count = len(points)

        if self.config.highlight_heads:
            size = int(np.floor(10 - np.log10(1 + count)))
            if size < 1:
                size = 1
            for x, y in points:
                cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)

        self.history[self.counter % self.config.smoothing_queue_size] = count

        smoothed_count = int(np.nanmean(self.history))

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, str(smoothed_count), (100, 100), font, 4, (0, 255, 255), 10, cv2.LINE_AA)

        logger.info("Found %d persons. Smoothed count is %d", count, smoothed_count)
        # print("Found %d persons. Smoothed count is %d".format(count, smoothed_count))
        self.counter += 1

        return frame
