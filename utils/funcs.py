import logging
import os
from pathlib import Path

import yaml
from marshmallow_dataclass import class_schema

from utils.params import ServiceParams, Mode


def read_configs():
    trackers = {}
    config_path = 'config.yaml'
    if Path(config_path).exists():
        # читаем yaml-конфиг
        with open(config_path, "r") as input_stream:
            schema = class_schema(ServiceParams)()
            service_params: ServiceParams = schema.load(yaml.safe_load(input_stream))
            use_models = os.environ.get("use_models", "")
            if not use_models:
                logging.warning("Environment variable `use_models` is not set. It will be read from `config.yaml`")
                use_models = service_params.use_models

            use_models = list(map(lambda x: x.lower().strip(), use_models.split(',')))

            for model in use_models:
                if model == "iim":
                    from detection_models.iim.misc.params import IimParams
                    from utils.iim_funcs import IimModel

                    iim_config_path = f'detection_models/{model}/config.yaml'
                    if Path(iim_config_path).exists():
                        with open(iim_config_path, "r") as stream:
                            schema = class_schema(IimParams)()
                            params: IimParams = schema.load(yaml.safe_load(stream))

                            trackers[Mode.IIM] = IimModel(params)
                    else:
                        logging.error("File '%s' doesn't exist", iim_config_path)
                elif model == "bytetrack":
                    from utils.bytetrack_funcs import BytetrackModel
                    from utils.bytetrack_params import BytetrackParams

                    bytetrack_config_path = f'detection_models/{model}/config.yaml'
                    if Path(bytetrack_config_path).exists():
                        with open(bytetrack_config_path, "r") as stream:
                            schema = class_schema(BytetrackParams)()
                            params: BytetrackParams = schema.load(yaml.safe_load(stream))
                            trackers[Mode.BYTETRACK] = BytetrackModel(params)
                    else:
                        logging.error("File '%s' doesn't exist", bytetrack_config_path)
                else:
                    raise ValueError(f"{model} model doesn't supported. "
                                     f"The only options available are 'bytetrack' and 'iim'.")
    else:
        logging.error("File '%s' doesn't exist", config_path)
        raise ValueError("Config file is absent")

    return service_params, trackers
