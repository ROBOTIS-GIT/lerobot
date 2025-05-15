# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import math
import time
import traceback
from copy import deepcopy
import threading

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import RosMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import uuid

class JointTrajectoryPublisher(Node):
    def __init__(self, topic_name, node_name=None):
        if node_name is None:
            node_name = f"joint_trajectory_publisher_{uuid.uuid4().hex[:8]}"
        super().__init__(node_name)

        self.publisher = self.create_publisher(JointTrajectory, topic_name, 10)

    def publish(self, joint_trajectory: JointTrajectory):
        self.publisher.publish(joint_trajectory)


class JointStatesSubscriber(Node):
    def __init__(self, callback, topic_name="/joint_states", node_name=None, fps=None):
        if node_name is None:
            node_name = f"joint_states_subscriber_{uuid.uuid4().hex[:8]}"
        super().__init__(node_name)
        self.callback = callback
        self.fps = fps
        self.latest_msg = None
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            JointState,
            topic_name,
            self.listener_callback,
            10
        )

        if self.fps is not None:
            self.timer = self.create_timer(1.0 / self.fps, self.process_message)

    def listener_callback(self, msg: JointState):
        with self.lock:
            self.latest_msg = msg
        if self.fps is None:
            joint_data = {
                "name": list(msg.name),
                "position": np.array(msg.position),
                "velocity": np.array(msg.velocity),
                "effort": np.array(msg.effort),
                "timestamp_utc": capture_timestamp_utc(),
            }
            self.callback(joint_data)

    def process_message(self):
        with self.lock:
            if self.latest_msg is None:
                return
            msg = self.latest_msg
            self.latest_msg = None
        joint_data = {
            "name": list(msg.name),
            "position": np.array(msg.position),
            "velocity": np.array(msg.velocity),
            "effort": np.array(msg.effort),
            "timestamp_utc": capture_timestamp_utc(),
        }
        self.callback(joint_data)


class JointTrajectorySubscriber(Node):
    def __init__(self, callback, topic_name="/joint_trajectory", node_name=None, fps=None):
        if node_name is None:
            node_name = f"joint_trajectory_subscriber_{uuid.uuid4().hex[:8]}"
        super().__init__(node_name)
        self.callback = callback
        self.fps = fps
        self.latest_msg = None
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            JointTrajectory,
            topic_name,
            self.listener_callback,
            10
        )

        if self.fps is not None:
            self.timer = self.create_timer(1.0 / self.fps, self.process_message)

    def listener_callback(self, msg: JointTrajectory):
        with self.lock:
            self.latest_msg = msg
        if self.fps is None:
            if not msg.points:
                return
            point = msg.points[0]
            joint_data = {
                "name": list(msg.joint_names),
                "position": np.array(point.positions) if point.positions else np.zeros(len(msg.joint_names)),
                "velocity": np.array(point.velocities) if point.velocities else np.zeros(len(msg.joint_names)),
                "effort": np.array(point.effort) if point.effort else np.zeros(len(msg.joint_names)),
                "timestamp_utc": capture_timestamp_utc(),
            }
            self.callback(joint_data)

    def process_message(self):
        with self.lock:
            if self.latest_msg is None:
                return
            msg = self.latest_msg
            self.latest_msg = None
        if not msg.points:
            return
        point = msg.points[0]
        joint_data = {
            "name": list(msg.joint_names),
            "position": np.array(point.positions) if point.positions else np.zeros(len(msg.joint_names)),
            "velocity": np.array(point.velocities) if point.velocities else np.zeros(len(msg.joint_names)),
            "effort": np.array(point.effort) if point.effort else np.zeros(len(msg.joint_names)),
            "timestamp_utc": capture_timestamp_utc(),
        }
        self.callback(joint_data)

def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key

def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name

class RosMotorsBus:
    """
    The RosMotorsBus class allows to efficiently read and write through ROS2 topics.
    A RosMotorsBus instance requires a topic_name (e.g. `RosMotorsBus(topic_name="/joint_states")`).

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = ""

    config = RosMotorsBusConfig(
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus = RosMotorsBus(config)
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        config: RosMotorsBusConfig,
    ):
        self.motors = config.motors
        self.mock = config.mock

        self.topic_name = config.topic_name
        self.topic_type = config.topic_type
        self.action_topic_name = config.action_topic_name
        self.fps = config.fps

        self.is_connected = True
        self.logs = {}
        self.latest_joint_state = None  # Initialize as empty dict
        self._ros_subscriber_node = None
        self._ros_publisher_node = None
        self._ros_nodes = None

    def _wait_for_joint_state(self):
        while self.latest_joint_state is None:
            time.sleep(1)
            print("No joint state yet... waiting")

    def connect(self):
        # Instead of starting a thread, just create the node
        self._run_ros_node()
        self.is_connected = True
        self._ros_nodes = [self._ros_subscriber_node]
        if self.action_topic_name is not None:
            self._ros_nodes.append(self._ros_publisher_node)

    def _run_ros_node(self):
        # Create the node but do not spin it
        if self.topic_type is JointState:
            self._ros_subscriber_node = JointStatesSubscriber(self._joint_callback, topic_name=self.topic_name, fps=self.fps)
        elif self.topic_type is JointTrajectory:
            self._ros_subscriber_node = JointTrajectorySubscriber(self._joint_callback, topic_name=self.topic_name, fps=self.fps)
        else:
            raise ValueError(f"Unknown topic type: {self.topic_type}")

        if self.action_topic_name is not None:
            self._ros_publisher_node = JointTrajectoryPublisher(self.action_topic_name)

    def _joint_callback(self, joint_data: dict):
        timestamp = joint_data["timestamp_utc"]
        joint_name_to_index = {name: i for i, name in enumerate(joint_data["name"])}
        for name in self.ros_joint_names:
            if name in joint_name_to_index:
                idx = joint_name_to_index[name]
                if self.latest_joint_state is None:
                    self.latest_joint_state = {}
                self.latest_joint_state[name] = {
                    "position": joint_data["position"][idx],
                    "velocity": joint_data["velocity"][idx]
                }
                self.logs[f"timestamp_utc_{name}"] = timestamp

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("RosMotorsBus is not connected.")

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names
        if isinstance(motor_names, str):
            motor_names = [motor_names]

        values = []
        for name in motor_names:
            ros_joint_name = self.motors[name][1]
            if ros_joint_name not in self.latest_joint_state:
                raise ValueError(f"No joint state received for {ros_joint_name}")

            if data_name == "Present_Position":
                values.append(self.latest_joint_state[ros_joint_name]["position"])
            elif data_name == "Present_Velocity":
                values.append(self.latest_joint_state[ros_joint_name]["velocity"])
            else:
                raise ValueError(f"Unknown data name: {data_name}")

        values = np.array(values, dtype=np.float32)

        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def reconnect(self):
        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def ros_joint_names(self) -> list[str]:
        return [ros_joint_name for _, ros_joint_name in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        ros_joint_names = []
        for name in motor_names:
            motor_idx, ros_joint_name = self.motors[name]
            motor_ids.append(motor_idx)
            ros_joint_names.append(ros_joint_name)

        values = values.tolist()

        if self.action_topic_name is not None:
            self._ros_publisher_node.publish(
                JointTrajectory(joint_names=ros_joint_names,
                                points=[JointTrajectoryPoint(positions=values, joint_names=ros_joint_names)]))

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def get_ros_nodes(self):
        return self._ros_nodes
