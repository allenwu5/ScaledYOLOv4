FROM openvino/ubuntu20_dev:2021.3

USER root

# https://askubuntu.com/a/1013396
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Python3.7 for Tensorflow 1
RUN apt-get install -y python3.7
RUN /usr/bin/python3.7 -m pip install tensorflow-cpu==1.15 Pillow numpy==1.19

# https://github.com/TNTWEN/OpenVINO-YOLOV4/
# yolov4-tiny.weights to frozen_darknet_yolov4_model.pb
# OpenVINO-YOLOV4/convert_weights_pb.py --class_names=data/coco.names --weights_file=yolov4-tiny.weights --data_format=NHWC --tiny --size=608 