import yaml
from marshmallow_dataclass import class_schema

from detection_models.iim.misc.params import IimParams
from utils.bytetrack_funcs import BytetrackModel
from utils.bytetrack_params import BytetrackParams
from utils.iim_funcs import IimModel


def test_initialize_iim():
    params = IimParams(weight_path="./detection_models/iim/weights/FDST-HR.pth")
    model = IimModel(params)
    assert model.model is not None


def test_initialize_bytetrack():
    bytetrack_config_path = 'detection_models/bytetrack/config.yaml'
    with open(bytetrack_config_path, "r") as stream:
        schema = class_schema(BytetrackParams)()
        params: BytetrackParams = schema.load(yaml.safe_load(stream))
        # на основе параметров инитим модель
        tracker = BytetrackModel(params)
    assert tracker is not None


test_initialize_iim()
test_initialize_bytetrack()
