FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get update -y && apt-get -y upgrade && \
	apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get update -y && apt-get install -y python3-pip python3-dev && \
	apt-get update -y && apt-get install -y python3 && \
	apt-get update -y && apt-get install -y git


RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN mkdir online_tracker
COPY detection_models/bytetrack/requirements.txt online_tracker/detection_models/bytetrack/requirements.txt
RUN pip install -r online_tracker/detection_models/bytetrack/requirements.txt

COPY detection_models/iim/requirements.txt online_tracker/detection_models/iim/requirements.txt
RUN pip install -r online_tracker/detection_models/iim/requirements.txt

COPY requirements.txt online_tracker/requirements.txt
RUN pip install -r online_tracker/requirements.txt

RUN pip install cython
RUN pip install cython_bbox
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN sed -i "s/self._ydl_info\['like_count'\]/0/g" /usr/local/lib/python3.8/dist-packages/pafy/backend_youtube_dl.py
RUN sed -i "s/self._ydl_info\['dislike_count'\]/0/g" /usr/local/lib/python3.8/dist-packages/pafy/backend_youtube_dl.py


COPY utils online_tracker/utils
COPY detection_models online_tracker/detection_models

# Bytrack section
WORKDIR /online_tracker/detection_models/bytetrack
RUN python3 setup.py develop
ENV PYTHONPATH="${PYTHONPATH}:/online_tracker/detection_models/bytetrack:/online_tracker/detection_models/bytetrack/yolox"
ENV PATH="./:${PATH}"

WORKDIR /online_tracker
COPY templates online_tracker/templates
COPY flask_app.py online_tracker/flask_app.py
COPY config.yaml online_tracker/config.yaml

ENTRYPOINT [ "python", "flask_app.py"]


