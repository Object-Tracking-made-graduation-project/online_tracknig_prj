import torch

from detection_models.bytetrack.tools.demo_track import Predictor
from detection_models.bytetrack.tools.demo_track import Predictor
from detection_models.bytetrack.yolox.exp import get_exp
from detection_models.bytetrack.yolox.utils import fuse_model, get_model_info, postprocess
from detection_models.bytetrack.yolox.tracker.byte_tracker import BYTETracker
from detection_models.bytetrack.yolox.tracking_utils.timer import Timer
from detection_models.bytetrack.yolox.utils.visualize import plot_tracking


class BytetrackModel:
    """
    Класс модели Bytetrack. Здесь его инициализируем и апдейтим при инференсе
    """
    def __init__(self, config):
        self.config = config.model_params
        self.detector, self.exp = get_predictor(config.model_params)
        self.model = BYTETracker(config.model_params, frame_rate=30)

    def online_inference(self, frame):
        """
        функция для инференса bytetrack
        """
        timer = Timer()
        outputs, img_info = self.detector.inference(frame, timer)
        frame_id = 0
        if outputs[0] is not None:
            online_targets = self.model.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.config.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.config.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
        return online_im


def get_predictor(args):
    """
    функция получения предиктора
    """
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model().to(args.device)
    model.eval()
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    model = model.half()
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    return predictor, exp