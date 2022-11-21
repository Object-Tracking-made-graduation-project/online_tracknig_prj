import os
from collections import OrderedDict

import torch

from ..misc.params import IimParams
from ..model.locator import Crowd_locator


def load_model(model_params: IimParams):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(model_params.gpu_id)

    net = Crowd_locator(model_params.netName, model_params.gpu_id, pretrained=True)
    net.to(device)
    state_dict = torch.load(model_params.weight_path, map_location=device)
    if len(model_params.gpu_id.split(',')) > 1:
        net.load_state_dict(state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()

    return net
