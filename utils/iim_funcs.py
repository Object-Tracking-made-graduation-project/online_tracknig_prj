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
        self.frame_counter = 0

    def online_inference(self, frame: np.ndarray, video_mask: np.array) -> np.ndarray:
        points = get_points_on_image(frame, self.model)
        count = len(points)

        size = int(np.floor(10 - np.log10(1 + count)))
        if size < 1:
            size = 1
        head_counter = 0
        for x, y in points:
            # если задана маска то проверяем, находится ли точка в выделенном сегменте
            # если находится, то рисуем, иначе нет
            if video_mask is not None:
                if video_mask[int(y), int(x)][3] < 155:
                    head_counter += 1
                    if self.config.highlight_heads:
                        cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)
            else:
                head_counter += 1
                if self.config.highlight_heads:
                    cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)

        self.history[self.frame_counter % self.config.smoothing_queue_size] = head_counter
        smoothed_count = int(np.nanmean(self.history))

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, str(smoothed_count), (100, 100), font, 4, (0, 255, 255), 10, cv2.LINE_AA)
        # здесь накладываем маску на основную картинку, чтобы выделить фрагмент и затенить остальное
        if video_mask is not None:
            trans_mask = video_mask[:, :, 3] == 0
            video_mask[trans_mask] = [105, 105, 105, 255]
            # video_mask[trans_mask] = [224, 11, 161, 255]
            new_mask = cv2.cvtColor(video_mask, cv2.COLOR_BGRA2BGR)
            frame = cv2.addWeighted(frame, 1.0, new_mask, 0.8, 0.0)

        self.frame_counter += 1
        logging.info("Total found: %d, after mask: %d, smoothed: %d", count, head_counter, smoothed_count)

        return frame
    