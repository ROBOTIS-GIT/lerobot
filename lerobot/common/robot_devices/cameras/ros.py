import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lerobot.common.robot_devices.cameras.configs import RosCameraConfig
from lerobot.common.utils.utils import capture_timestamp_utc

class ROSImageSubscriber(Node):
    def __init__(self, image_callback, stop_event: threading.Event, topic_name="/camera/camera/color/image_rect_raw", fps=30):
        super().__init__("ros_image_subscriber")
        self.bridge = CvBridge()
        self.image_callback = image_callback
        self.stop_event = stop_event
        self.fps = fps

        self.latest_msg = None
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10,
        )

        # Convert only at given FPS
        self.timer = self.create_timer(1.0 / self.fps, self.process_image)

    def listener_callback(self, msg):
        with self.lock:
            self.latest_msg = msg

    def process_image(self):
        if self.stop_event.is_set():
            rclpy.shutdown()
            return

        with self.lock:
            if self.latest_msg is None:
                return
            msg = self.latest_msg
            self.latest_msg = None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.image_callback(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


class RosCamera:
    def __init__(self, config: RosCameraConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.channels = config.channels
        self.fps = config.fps
        self.topic_name = config.topic_name

        self.color_image = None
        self.logs = {}
        self.is_connected = False
        self.stop_event = threading.Event()
        self.ros_node = None

    def _wait_for_image(self):
        while self.color_image is None:
            time.sleep(1)
            print("No image yet... waiting")

    def connect(self):
        self.is_connected = True
        self._init_ros_node()

    def _init_ros_node(self):
        self.ros_node = ROSImageSubscriber(self._ros_image_callback, self.stop_event, topic_name=self.topic_name, fps=self.fps)

    def _ros_image_callback(self, image_np: np.ndarray):
        self.color_image = image_np
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.logs["delta_timestamp_s"] = 1 / self.fps

    # def read(self) -> np.ndarray:
    #     start_time = time.perf_counter()
    #     self.logs["timestamp_utc"] = capture_timestamp_utc()
    #     self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

    #     if self.color_image is None:
    #         return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #     return self.color_image

    def async_read(self) -> np.ndarray:
        while True:
            if self.color_image is not None:
                return self.color_image

    def disconnect(self):
        self.is_connected = False
        if self.stop_event:
            self.stop_event.set()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def get_ros_node(self):
        return self.ros_node
