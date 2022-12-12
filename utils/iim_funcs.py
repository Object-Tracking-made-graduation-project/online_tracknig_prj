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

    def online_inference(self, frame: np.ndarray, video_mask: np.array) -> np.ndarray:
        points = get_points_on_image(frame, self.model)
        count = len(points)
        size = int(np.floor(10 - np.log10(1 + count)))
        if size < 1:
            size = 1
        counter = 0
        for x, y in points:
            # если задана маска то проверяем, находится ли точка в выделенном сегменте
            # если находится, то рисуем, иначе нет
            if video_mask is not None:
                if int(y) < video_mask.shape[0] and int(x) < video_mask.shape[1]:
                    if video_mask[int(y), int(x)][3] < 155:
                        counter += 1
                        cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)
            else:
                counter += 1
                cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, str(counter), (100, 100), font, 4, (0, 255, 255), 10, cv2.LINE_AA)
        # здесь накладываем маску на основную картинку, чтобы выделить фрагмент и затонить остальное
        if video_mask is not None:
            tmp_mask = video_mask.copy()
            trans_mask = tmp_mask[:, :, 3] == 0
            tmp_mask[trans_mask] = [105, 105, 105, 255]
            new_mask = cv2.cvtColor(tmp_mask, cv2.COLOR_BGRA2BGR)
            frame = cv2.addWeighted(frame, 1.0, new_mask, 0.8, 0.0)

        logging.info("Found %d persons", counter)
        print(f"Found {counter} persons", counter)

        return frame
