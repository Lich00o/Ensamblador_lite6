"""Detector ChArUco -> TF (frame óptico -> charuco_board) + PoseStamped.

Solo publica detecciones que cumplen min_corners y max_reproj_rmse_px.
"""

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from .charuco import BoardConfig, CharucoPoseEstimator
from .image_conv import bgr_to_msg, image_to_numpy, to_bgr, to_gray
from .transforms import fill_ros_pose, fill_ros_transform


class CharucoDetectorNode(Node):
    def __init__(self):
        super().__init__('charuco_detector')
        p = self.declare_parameter
        p('board.squares_x', 5)
        p('board.squares_y', 7)
        p('board.square_length', 0.030)
        p('board.marker_length', 0.022)
        p('board.dictionary', 'DICT_4X4_50')
        p('image_topic', '/zed/zed_node/rgb/color/rect/image')
        p('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')
        p('board_frame', 'charuco_board')
        p('publish_tf', True)
        p('publish_debug_image', True)
        p('min_corners', 8)
        p('max_reproj_rmse_px', 1.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cfg = BoardConfig(g('board.squares_x'), g('board.squares_y'),
                               g('board.square_length'), g('board.marker_length'),
                               g('board.dictionary'))
        self.estimator = CharucoPoseEstimator(self.cfg)
        self.board_frame = g('board_frame')
        self.min_corners = g('min_corners')
        self.max_err = g('max_reproj_rmse_px')
        self.publish_tf = g('publish_tf')
        self.K = None
        self.D = None

        self.tf_pub = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '~/board_pose', 10)
        self.debug_pub = (self.create_publisher(Image, '~/debug_image', 1)
                          if g('publish_debug_image') else None)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info, 10)
        self.create_subscription(Image, g('image_topic'), self._on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f'ChArUco {self.cfg.squares_x}x{self.cfg.squares_y} sq={self.cfg.square_length} '
            f'mk={self.cfg.marker_length} {self.cfg.dictionary}; imagen: {g("image_topic")}')

    def _on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
        if msg.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
            self.get_logger().warn(f'modelo de distorsión {msg.distortion_model} no soportado',
                                   throttle_duration_sec=10.0)

    def _on_image(self, msg):
        if self.K is None:
            self.get_logger().warn('esperando camera_info', throttle_duration_sec=5.0)
            return
        try:
            img, enc = image_to_numpy(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        det = self.estimator.estimate(to_gray(img, enc), self.K, self.D, self.min_corners)
        ok = det is not None and det.reproj_rmse_px <= self.max_err
        if ok:
            pose = PoseStamped()
            pose.header = msg.header
            fill_ros_pose(pose.pose, det.T_cam_board)
            self.pose_pub.publish(pose)
            if self.publish_tf:
                tf = TransformStamped()
                tf.header = msg.header
                tf.child_frame_id = self.board_frame
                fill_ros_transform(tf.transform, det.T_cam_board)
                self.tf_pub.sendTransform(tf)
        if self.debug_pub is not None and self.debug_pub.get_subscription_count() > 0:
            bgr = to_bgr(img, enc)
            if det is not None:
                self.estimator.draw(bgr, det, self.K, self.D)
            if not ok:
                cv2.putText(bgr, 'RECHAZADA' if det is not None else 'SIN TABLERO', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.debug_pub.publish(bgr_to_msg(bgr, msg.header, Image))


def main():
    rclpy.init()
    node = CharucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
