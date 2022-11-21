from dataclasses import dataclass

from utils.params import ModelParams

model_path = r'./FDST-HR.pth'


@dataclass()
class IimParams(ModelParams):
    weight_path: str
    gpu_id: str = '0'
    netName: str = 'HR_Net'


@dataclass()
class ImageParams:
    path: str = None
    output_dir: str = None


@dataclass()
class InferenceParams:
    model_params: ModelParams
    image_params: ImageParams = None
