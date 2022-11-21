from dataclasses import dataclass, field

URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"


@dataclass()
class ModelParams:
    pass


@dataclass()
class ServiceParams:
    use_model: str = "iim"
    video_url: str = field(default=URL_MODEL)
    stream: int = 0
    model_params: ModelParams = None
