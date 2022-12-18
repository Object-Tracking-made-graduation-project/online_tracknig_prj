import asyncio
import logging
import os
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import pafy
from flask import Flask, render_template, Response, request

from utils.funcs import read_configs
from utils.params import ServiceParams, Mode

app = Flask(__name__)

VIDEO_PATH = 'D:\\pycharm_projects\\made\\diploma\\docker_jupyter\\' \
             '1 эт. Б AromaCafe Зал 1_20221024-160000--20221024-160500.avi'

VIDEO_MASK = None

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

trackers = dict()
reset = True


def get_video_obj(url, stream=-1):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object.
    """
    if not os.path.exists(url):
        pafy_obj = pafy.new(url)

        if stream >= len(pafy_obj.streams):
            logger.warning("Stream %d doesn't exist. It will be used as -1.")
            stream = -1
        play = pafy_obj.streams[stream]
        assert play is not None, f"Can't extract videostream from {url}"
        video_obj = cv2.VideoCapture(play.url)
        return video_obj
    else:
        video_obj = cv2.VideoCapture(url)
        return video_obj


async def get_frame():
    ret_val, frame = video_capture.read()
    return frame


def generate_frames():  # generate frame by frame from camera
    global real_time_s, video_time_s, video_capture, reset
    loop = asyncio.new_event_loop()
    counter = 0
    while True:
        if video_capture is not None:
            if current_mode == 1:
                # объект определили заранее, теперь читаем кадр и отправляем на веб морду
                try:
                    frame = loop.run_until_complete(get_frame())
                    if frame is not None:
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                except Exception as e:
                    print("catched")

            else:
                interval = service_params.interval.get(current_mode, 0.)
                if counter == service_params.frames_num_before_show:
                    counter = 0

                    cur_time_s = time.time()
                    diff_s = cur_time_s - real_time_s
                    #if diff_s >= interval:
                    if video_capture.isOpened():
                        try:
                            time_s = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                            if time_s - video_time_s < interval:
                                continue
                            frame = loop.run_until_complete(get_frame())
                            #success, frame = loop.run_until_complete(get_frame())
                            if frame is not None:
                                success = True
                            else:
                                success = False
                            if success:
                                tracker = trackers.get(current_mode, None)
                                if tracker is not None:
                                    if reset:
                                        tracker.reset()
                                        reset = False
                                    frame = tracker.online_inference(frame, VIDEO_MASK)
                                ret, buffer = cv2.imencode('.jpg', frame)
                                frame = buffer.tobytes()
                                real_time_s = cur_time_s
                                video_time_s = time_s
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                        except Exception as e:
                            logger.error(str(e))
                    else:
                        logger.warning('videocapture is closed. Trying to reopen it.')
                        video_capture = get_video_obj(video_url)
                    #else:
                        #if interval - diff_s < 30.:
                            #time.sleep(interval - diff_s)
                else:
                    counter += 1
        time.sleep(0.001)


@app.route('/ping')
def ping():
    """Video streaming home page."""
    return Response("It works!")


@app.route('/load_mask', methods=["POST"])
def load_inference_mask():
    """
    Функция загрузки маски
    """
    global VIDEO_MASK
    try:
        img_path = parse_request_and_upload(request)
        VIDEO_MASK = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        tracker = trackers.get(current_mode, None)
        if tracker is not None:
            tracker.reset()

        response_message = f"mask loaded."
        logger.info(response_message)
        status = 200
    except Exception as e:
        logger.error("Error on mask load: %s", str(e))
        response_message = f"Error on mask load:\n {str(e)}"
        status = 503
    return Response(response_message, status=status)


@app.route('/clear_mask', methods=["POST"])
def clear_mask():
    """
    Функция очиски маски
    """
    global VIDEO_MASK
    VIDEO_MASK = None
    tracker = trackers.get(current_mode, None)
    if tracker is not None:
        tracker.reset()
    response_message = 'mask deleted'
    logger.info(response_message)
    status = 200
    return Response(response_message, status=status)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    generator = generate_frames()  # здесь используется не асинхронная версия
    # generator = gen_frames()   # здесь используется асинхронная версия
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_request_and_upload(request_obj):
    if len(request_obj.files) > 0:
        upload_path = Path(os.getcwd()) / UPLOAD_FOLDER_NAME
        if not upload_path.exists():
            os.makedirs(upload_path)
        f = request_obj.files['file']
        file_path = upload_path / f.filename
        f.save(file_path)
        logger.info("File saved to %s", str(file_path))
        return str(file_path)
    elif len(request_obj.json) > 0:
        url = request_obj.json['video_url']
        if 'youtube.com' not in url:
            raise ValueError("Only links to youtube are supported")
        return url


@app.route('/init', methods=["POST"])
def init():
    global video_url, video_capture, real_time_s, video_time_s, VIDEO_MASK, trackers, reset
    response_message = "Initializing..."
    status = 200
    try:
        reset = True
        tracker = trackers.get(current_mode, None)
        if tracker is not None:
            tracker.reset()
            logger.debug("%s tracker.reset()", tracker.name())
        video_url = parse_request_and_upload(request)
        print(f"{video_url} is being processed.")
        if video_capture is not None and video_capture.isOpened():
            video_capture.release()
        video_capture = get_video_obj(video_url)
        real_time_s = time.time()
        video_time_s = 0.
        if video_capture.isOpened():
            response_message = f"{video_url} has been initialized."
        else:
            raise ValueError(f"Can't open file {video_url}")

    except Exception as exc:
        logger.error("Error while init: %s", str(exc))
        response_message = f"Error while initializing {video_url}:\n {str(exc)}"
        status = 503

    return Response(response_message, status=status)


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

    service_params, trackers = read_configs()
    app.run(host='0.0.0.0', port=8132)
