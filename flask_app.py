import asyncio
import logging
import os
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from multiprocessing.pool import ThreadPool

import cv2
import pafy
from flask import Flask, render_template, Response, request

from utils.funcs import read_configs
from utils.params import ServiceParams, Mode

app = Flask(__name__)

VIDEO_PATH = 'D:\\pycharm_projects\\made\\diploma\\docker_jupyter\\' \
             '1 эт. Б AromaCafe Зал 1_20221024-160000--20221024-160500.avi'


pool = ThreadPool(processes=1)

service_params: ServiceParams = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER_NAME = 'uploads'
video_url = ''

video_capture: cv2.VideoCapture = None
service_params: ServiceParams = None

video_time_s: float = None
real_time_s: float = None
current_mode: Mode = Mode.ORIGINAL


def get_video_obj(url, stream=-1):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """
    if not os.path.exists(url):
        play = pafy.new(url).streams[stream]
        assert play is not None
        video_obj = cv2.VideoCapture(play.url)
        return video_obj
    else:
        video_obj = cv2.VideoCapture(url)
        return video_obj


async def get_frame():
    ret_val, frame = video_capture.read()
    return frame


def generate_frames():  # generate frame by frame from camera
    global real_time_s, video_time_s, video_capture
    loop = asyncio.new_event_loop()
    while True:

        if video_capture is not None:
            # fps = video_capture.get(cv2.CAP_PROP_FPS)

            interval = service_params.interval.get(current_mode, 0.)
            if interval <= 0:
                logger.warning("interval %f s", interval)

            cur_time_s = time.time()
            diff_s = cur_time_s - real_time_s
            logger.debug("diff_s=%f, cur_time_s=%f, real_time_s=%f", diff_s, cur_time_s, real_time_s)
            if diff_s >= interval:
                if video_capture.isOpened():
                    try:

                        grabbed = video_capture.grab()
                        if grabbed:
                            time_s = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                            if time_s - video_time_s < interval:
                                continue

                            logger.debug("time_s %f", time_s)
                            frame = loop.run_until_complete(get_frame())
                            #success, frame = video_capture.retrieve()
                            success = True
                            if success:
                                tracker = service_params.trackers.get(current_mode, None)
                                if tracker is not None:
                                    frame = tracker.online_inference(frame)
                                ret, buffer = cv2.imencode('.jpg', frame)
                                frame = buffer.tobytes()
                                real_time_s = cur_time_s
                                video_time_s = time_s
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                        else:
                            logger.warning("not grabbed")
                    except Exception as e:
                        logger.error(str(e))
                else:
                    logger.warning('videocapture is closed')
                    video_capture = get_video_obj(video_url)
            else:
                if interval - diff_s < 30.:
                    time.sleep(interval - diff_s)
        time.sleep(0.001)


def gen_frames():
    """
    здесь получаем кадр с потока, получаем по нему инференс с модели и отправляем на форму
    """
    global video_capture
    counter = 0
    #if not video_capture:
        #raise ValueError("video_capture is empty")

    loop = asyncio.new_event_loop()
    while True:
        if video_capture is not None:
            # while video_capture.isOpened():
            # объект определили заранее, теперь читаем кадр и отправляем в модель
            frame = loop.run_until_complete(get_frame())
            if counter == service_params.frames_num_before_show:
                counter = 0
                # перед отправкой на модель меняем размер кадра.
                # здесь мы получаем инференс
                tracker = service_params.trackers.get(current_mode, None)
                if tracker is not None:
                    out_frame = tracker.online_inference(frame)
                else:
                    out_frame = frame
                ret, buffer = cv2.imencode('.jpg', out_frame)
                out_frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')  # concat frame one by one and show result
            else:
                counter += 1


@app.route('/ping')
def ping():
    """Video streaming home page."""
    return Response("It works!")


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    generator = generate_frames()   # здесь используется не асинхронная версия
    #generator = gen_frames()   # здесь используется асинхронная версия
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_request(request_obj):
    if len(request_obj.files) > 0:
        upload_path = f'{os.path.dirname(sys.argv[0])}/{UPLOAD_FOLDER_NAME}'
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        f = request_obj.files['file']
        fname = f.filename
        upload_path = f'{upload_path}/{fname}'
        f.save(upload_path)
        return upload_path
    elif len(request_obj.json) > 0:
        return request_obj.json['video_url']


@app.route('/init', methods=["POST"])
def init():
    global video_url, video_capture, real_time_s, video_time_s
    video_url = parse_request(request)
    #video_url = request.json['video_url']
    print(f"{video_url} is being processed.")
    if video_capture is not None:
        video_capture.release()
    video_capture = get_video_obj(video_url)
    real_time_s = time.time()
    video_time_s = 0.
    return Response(f"{video_url} has been initialized")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/set_mode/<mode>', methods=['GET', 'POST'])
def set_mode(mode: str):
    global current_mode
    current_mode = int(mode)
    logger.debug("current_mode is %d", current_mode)
    return mode


if __name__ == '__main__':
    print("Flask app started.")

    service_params = read_configs()
    app.run(host='0.0.0.0', port=8132)
