"""Conversión sensor_msgs/Image <-> numpy sin cv_bridge."""

import cv2
import numpy as np

_CHANNELS = {'mono8': 1, 'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4,
             '8UC1': 1, '8UC3': 3, '8UC4': 4}

_TO_GRAY = {'rgb8': cv2.COLOR_RGB2GRAY, 'bgr8': cv2.COLOR_BGR2GRAY, '8UC3': cv2.COLOR_BGR2GRAY,
            'rgba8': cv2.COLOR_RGBA2GRAY, 'bgra8': cv2.COLOR_BGRA2GRAY, '8UC4': cv2.COLOR_BGRA2GRAY}

_TO_BGR = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR, 'bgra8': cv2.COLOR_BGRA2BGR,
           '8UC4': cv2.COLOR_BGRA2BGR, 'mono8': cv2.COLOR_GRAY2BGR, '8UC1': cv2.COLOR_GRAY2BGR}


def image_to_numpy(msg):
    enc = msg.encoding
    if enc in ('yuv422', 'uyvy', 'yuv422_yuy2', 'yuyv'):
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = buf[:, :msg.width * 2].reshape(msg.height, msg.width, 2)
        code = cv2.COLOR_YUV2BGR_UYVY if enc in ('yuv422', 'uyvy') else cv2.COLOR_YUV2BGR_YUY2
        return cv2.cvtColor(img, code), 'bgr8'
    if enc not in _CHANNELS:
        raise ValueError(f'encoding no soportado: {enc}')
    ch = _CHANNELS[enc]
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width) if ch == 1 else img.reshape(msg.height, msg.width, ch)
    return img, enc


def to_gray(img, enc):
    return img if enc in ('mono8', '8UC1') else cv2.cvtColor(img, _TO_GRAY[enc])


def to_bgr(img, enc):
    return img.copy() if enc in ('bgr8', '8UC3') else cv2.cvtColor(img, _TO_BGR[enc])


def bgr_to_msg(bgr, header, msg_cls):
    msg = msg_cls()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr).tobytes()
    return msg
