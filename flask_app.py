import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from pathlib import Path

import cv2
import pafy
import yaml
from flask import Flask, render_template, Response
from marshmallow_dataclass import class_schema
from utils.params import ServiceParams
import time
import asyncio
import numpy as np

tracker = None

from threading import Thread
from queue import Queue
from multiprocessing.pool import ThreadPool

app = Flask(__name__)

VIDEO_PATH = 'D:\\pycharm_projects\\made\\diploma\\docker_jupyter\\' \
             '1 эт. Б AromaCafe Зал 1_20221024-160000--20221024-160500.avi'

raw_video_obj = None
det_video_obj = None

pool = ThreadPool(processes=1)

service_params: ServiceParams = None


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.frame = 0
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, daemon=True, args=())
        # t = Thread(target=self.update, daemon=True, args=())
        # t.daemon = True
        t.start()
        # t.join()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
                # self.frame = frame

    def read(self):
        # return next frame in the queue
        return 0, self.Q.get()
        # return self.frame

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def get_video_obj(url, stream=0):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """
    if not os.path.exists(url):
        play = pafy.new(url).streams[stream]
        assert play is not None
        # video_obj = FileVideoStream(play.url).start()
        # video_obj = FileVideoStream(VIDEO_PATH).start()
        video_obj = cv2.VideoCapture(play.url)
        # video_obj = cv2.VideoCapture(VIDEO_PATH)

        return video_obj
    else:
        # video_obj = FileVideoStream(url).start()
        # video_obj = FileVideoStream(VIDEO_PATH).start()
        video_obj = cv2.VideoCapture(url)
        # video_obj = cv2.VideoCapture(VIDEO_PATH)
        return video_obj


def gen_frames():  # generate frame by frame from camera
    if not raw_video_obj:
        raise ValueError("raw_video_obj is empty")
    while raw_video_obj.isOpened():
        success, frame = raw_video_obj.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


async def get_frame():
    ret_val, frame = det_video_obj.read()
    return frame

def gen_det_frames():  # generate frame by frame from camera
    """
    здесь получаем кадр с потока, получаем по нему инференс с модели и отправляем на форму
    """
    global det_video_obj
    counter = 0
    if not det_video_obj:
        raise ValueError("det_video_obj is empty")

    # while det_video_obj.more():
    loop = asyncio.new_event_loop()
    while det_video_obj.isOpened():
        # loop = asyncio.new_event_loop()
        # объект определили заранее, теперь читаем кадр и отправляем в модель
        #frame_start = time.time()
        # frame = det_video_obj.read()
        ret_val, frame = det_video_obj.read()
        #print(f'OPENCV elapsed time: {time.time() - frame_start}')
        if counter == service_params.frames_num_before_show:
            counter = 0
            # перед отправкой на модель меняем размер кадра.
            # здесь мы получаем инференс
            inf_start = time.time()
            frame = loop.run_until_complete(get_frame())
            # out_frame = loop.run_until_complete(tracker.online_inference(frame))
            # loop.close()
            out_frame = tracker.online_inference(frame)
            #print(f'inference elapsed time: {time.time() - inf_start}')
            det_ret, det_buffer = cv2.imencode('.jpg', out_frame)
            out_frame = det_buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')  # concat frame one by one and show result
        else:
            counter += 1


# def gen_frames():  # generate frame by frame from camera
#     while True:
#         # Capture frame-by-frame
#         success, frame = raw_video_obj.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# @app.route('/raw_video_feed')
# def raw_video_feed():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/det_video_feed')
def det_video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_det_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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


def init_video_objects():
    """
    парсинг аргументов командной строки
    """
    global raw_video_obj
    global det_video_obj
    # получаем видеопоток
    # raw_video_obj = get_video_obj(service_params.video_url, service_params.stream)
    det_video_obj = get_video_obj(service_params.video_url, service_params.stream)
    # et_video_obj = FileVideoStream(service_params.video_url, service_params.stream)#.start()
    # det_video_obj.set(cv2.CAP_PROP_FPS, 30)
    print("init_video_objects")


if __name__ == '__main__':
    print("Flask app started.")
    parser = ArgumentParser(
        prog="Test module for human detection",
        description="Test module for human detection",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    parser.set_defaults()
    # arguments.callback(arguments)

    default_config_path = 'config.yaml'
    if Path(default_config_path).exists():
        # читаем yaml-конфиг
        with open(default_config_path, "r") as input_stream:
            schema = class_schema(ServiceParams)()
            service_params: ServiceParams = schema.load(yaml.safe_load(input_stream))
            use_model = os.environ.get("use_model", "")
            if not use_model:
                print("Environment variable `use_model` is not set. It will be read from config.yaml")
                use_model = service_params.use_model

            if use_model == "bytetrack":
                from utils.bytetrack_funcs import BytetrackModel
                from utils.bytetrack_params import BytetrackParams

                bytetrack_config_path = 'detection_models/bytetrack/config.yaml'
                with open(bytetrack_config_path, "r") as stream:
                    schema = class_schema(BytetrackParams)()
                    params: BytetrackParams = schema.load(yaml.safe_load(stream))
                    # на основе параметров инитим модель
                    tracker = BytetrackModel(params)
            elif use_model == "iim":
                from detection_models.iim.misc.params import IimParams
                from utils.iim_funcs import IimModel

                iim_config_path = 'detection_models/iim/config.yaml'
                with open(iim_config_path, "r") as stream:
                    schema = class_schema(IimParams)()
                    params: IimParams = schema.load(yaml.safe_load(stream))
                    # на основе параметров инитим модель
                    tracker = IimModel(params)
                pass
            else:
                raise ValueError(f"{use_model} model doesn't supported. "
                                 f"The only options available are 'bytetrack' and 'iim'.")
            service_params.model_params = params

    init_video_objects()

    app.run(host='0.0.0.0', port=8132)
