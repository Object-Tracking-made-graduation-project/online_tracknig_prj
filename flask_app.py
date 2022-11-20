import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from pathlib import Path

import cv2
import pafy
import yaml
from flask import Flask, render_template, Response
from marshmallow_dataclass import class_schema


from utils.params import ServiceParams

tracker = None

app = Flask(__name__)

URL = "https://www.youtube.com/embed/2wqpy036z24"
URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"

VIDEO_PATH = 'D:\\pycharm_projects\\made\\diploma\\docker_jupyter\\ByteTrack-main\\' \
             '1 эт. Б AromaCafe Зал 1_20221024-160000--20221024-160500.avi'

raw_video_obj = ''
# raw_video_obj = get_video_obj(URL_MODEL)
# raw_video_obj.set(cv2.CAP_PROP_FPS, 30)
det_video_obj = ''
# det_video_obj = get_video_obj(URL_MODEL)
# det_video_obj.set(cv2.CAP_PROP_FPS, 30)

detector = ''
EXP = ''

params = ''


def get_video_obj(url):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """
    if not os.path.exists(url):
        play = pafy.new(url).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)
    else:
        return cv2.VideoCapture(url)


def gen_frames():
    """
    здесь получаем кадр с потока и отсылаем его на форму
    """
    while True:
        # объект определили заранее, теперь читаем кадр и отправляем на веб морду
        success, frame = raw_video_obj.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_det_frames():  # generate frame by frame from camera
    """
    здесь получаем кадр с потока, получаем по нему инференс с модели и отправляем на форму
    """
    while True:
        # объект определили заранее, теперь читаем кадр и отправляем в модель
        ret_val, frame = det_video_obj.read()
        # перед отправкой на модель меняем размер кадра.
        frame = cv2.resize(frame, (640, 480))
        # здесь мы получаем инференс
        out_frame = tracker.online_inference(frame)
        det_ret, det_buffer = cv2.imencode('.jpg', out_frame)
        out_frame = det_buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')  # concat frame one by one and show result


@app.route('/det_video_feed')
def det_video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_det_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/raw_video_feed')
# def raw_video_feed():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def setup_parser(parser):
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
                        help='list of options')
    parser.add_argument('-m', '--model',
                        dest='model',
                        default='bytetrack',
                        choices=['bytetrack', 'iim'],
                        help='change model for run',
                        required=False)
    parser.set_defaults()


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Test module for human detection",
        description="Test module for human detection",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)

    default_config_path = 'config.yaml'
    if Path(default_config_path).exists():
        # читаем yaml-конфиг
        with open(default_config_path, "r") as input_stream:
            schema = class_schema(ServiceParams)()
            service_params: ServiceParams = schema.load(yaml.safe_load(input_stream))

            if service_params.use_model == "bytetrack":
                from utils.bytetrack_funcs import BytetrackModel
                from utils.bytetrack_params import InferenceParams, BytetrackParams

                bytetrack_config_path = 'detection_models/bytetrack/config.yaml'
                with open(default_config_path, "r") as stream:
                    schema = class_schema(BytetrackParams)()
                    params: BytetrackParams = schema.load(yaml.safe_load(stream))
                    # на основе параметров инитим модель
                    tracker = BytetrackModel(params)
            elif service_params.use_model == "iim":
                from detection_models.iim.misc.params import IimParams
                from utils.iim_funcs import IimModel
                iim_config_path = 'detection_models/iim/config.yaml'
                with open(default_config_path, "r") as stream:
                    schema = class_schema(IimParams)()
                    params: IimParams = schema.load(yaml.safe_load(stream))
                    # на основе параметров инитим модель
                    tracker = IimModel(params)
                pass
            else:
                raise ValueError(f"{params.use_model} model doesn't supported. "
                                 f"The only options available are 'bytetrack' and 'iim'.")
            service_params.model_params = params

    #global det_video_obj
    # получаем видеопоток
    det_video_obj = get_video_obj(URL_MODEL)
    det_video_obj.set(cv2.CAP_PROP_FPS, 30)

    app.run(host='0.0.0.0', port=8132)
