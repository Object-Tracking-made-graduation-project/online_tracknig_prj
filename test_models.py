import yaml
from marshmallow_dataclass import class_schema

from detection_models.iim.misc.params import IimParams

from utils.iim_funcs import IimModel


def test_initialize_iim():
    params = IimParams(weight_path="./detection_models/iim/weights/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth")
    model = IimModel(params)
    assert model.model is not None


def test_initialize_bytetrack():
    from utils.bytetrack_funcs import BytetrackModel
    from utils.bytetrack_params import BytetrackParams
    bytetrack_config_path = 'detection_models/bytetrack/config.yaml'
    with open(bytetrack_config_path, "r") as stream:
        schema = class_schema(BytetrackParams)()
        params: BytetrackParams = schema.load(yaml.safe_load(stream))
        # на основе параметров инитим модель
        tracker = BytetrackModel(params)
    assert tracker is not None
