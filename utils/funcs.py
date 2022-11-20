import numpy as np

from utils.params import ModelParams

# TODO: make it abstract


class BaseModel:
    def __init__(self, model_params: ModelParams):
        pass

    def online_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        функция для инференса
        """
        pass