import pytest

from detection_models.iim.misc.params import IimModelParams
from utils.iim_funcs import IimModel


def test_initialize_iim():
    params = IimModelParams(weight_path="./detection_models/iim/FDST-HR.pth")
    model = IimModel(params)
    assert model.model is not None


test_initialize_iim()
