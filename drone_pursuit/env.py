import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import math

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=10,
)


class PursuitEnv(Node):
    def __init__(self, cfg, perception):
        super().__init__('pursuit_env')

        self.cfg = cfg
        self.perception = perception
        self.bridge = CvBridge()

        self.chaser_pos = np.zeros(3, dtype=np.float32)
        self.chaser_vel = np.zeros(3, dtype=np.float32)
        self.evader_pos = np.zeros(3, dtype=np.float32)

        self.vision_bit = 0
        self.bbox = None

        self._chaser_ready = False
        self._evader_ready = False

        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self._chaser_odom_cb,
            MAVROS_QOS
        )

        self.create_subscription(
            Odometry,
            '/evader/odom',
            self._evader_odom_cb,
            10
        )

        self.create_subscription(
            Image,
            '/iris/camera/image_raw',
            self._image_cb,
            10
        )

        self.get_logger().info('PursuitEnv node initialised')

    def _chaser_odom_cb(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear

        self.chaser_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.chaser_vel = np.array([v.x, v.y, v.z], dtype=np.float32)

        self._chaser_ready = True

    def _evader_odom_cb(self, msg):
        p = msg.pose.pose.position

        self.evader_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

        self._evader_ready = True

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            result = self.perception.detect(frame)

            self.vision_bit = result.vision_bit
            self.bbox = result.bbox

        except Exception as e:
            self.get_logger().warn(f'Image cb error: {e}')

    def is_ready(self):
        return self._chaser_ready and self._evader_ready

    def get_state(self):
        rel = self.evader_pos - self.chaser_pos
        dist = float(np.linalg.norm(rel))

        chaser_speed = float(np.linalg.norm(self.chaser_vel))

        if dist > 0.01 and chaser_speed > 0.01:
            dir_to_evader = rel / dist
            vel_norm = self.chaser_vel / chaser_speed
            dot = float(np.clip(np.dot(dir_to_evader, vel_norm), -1.0, 1.0))
            heading_err = float(math.acos(dot))
        else:
            heading_err = 0.0

        alt_err = float(self.evader_pos[2] - self.chaser_pos[2])

        base_state = np.array(
            [
                rel[0],
                rel[1],
                rel[2],
                self.chaser_vel[0],
                self.chaser_vel[1],
                self.chaser_vel[2],
                dist,
                heading_err,
                alt_err,
                float(self.vision_bit),
            ],
            dtype=np.float32
        )

        # YOLO bbox state:
        # bbox = (cx, cy, w, h, conf), all normalized.
        # If no detection, use zeros so state is always 15-D.
        if self.bbox is not None:
            bbox_state = np.array(self.bbox, dtype=np.float32)
        else:
            bbox_state = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

        state = np.concatenate([base_state, bbox_state]).astype(np.float32)

        return state

    def get_distance(self):
        return float(np.linalg.norm(self.evader_pos - self.chaser_pos))

    def compute_reward(self, prev_dist, curr_dist):
        rcfg = self.cfg['reward']
        dcfg = self.cfg['drone']

        # Safety termination
        if self.chaser_pos[2] < 1.0 or self.chaser_pos[2] > 6.0:
            return -100.0, True

        if np.linalg.norm(self.chaser_vel) > 4.0:
            return -100.0, True

        # Base distance penalty
        r = rcfg['step_dist_penalty'] * curr_dist

        # Main learning signal: reward moving closer
        progress_scale = rcfg.get('progress_reward_scale', 5.0)
        r += progress_scale * (prev_dist - curr_dist)

        # Vision reward
        if self.vision_bit == 1:
            r += 2.0
        else:
            r += rcfg['vision_penalty']

        # YOLO bbox shaping
        # bbox = (cx, cy, w, h, conf) all normalized 0-1
        if self.bbox is not None:
            cx, cy, bw, bh, conf = self.bbox
            center_error = abs(cx - 0.5) + abs(cy - 0.5)
            center_error = abs(float(cx) - 0.5) + abs(float(cy) - 0.5)

            # small weights only
            r -= 0.5 * center_error
            r += 1.0 * float(bw * bh)
            r += 0.2 * float(conf)
           

        # Proximity shaping
        if curr_dist < 5.0:
            r += rcfg['heading_bonus']

        if curr_dist < 2.0:
            r += rcfg['proximity_bonus']

        if curr_dist < 1.5:
            r += 5.0

        if curr_dist < 1.0:
            r += rcfg.get('close_bonus', 10.0)

        # Terminal success
        if curr_dist < dcfg['intercept_threshold']:
            return r + rcfg['terminal_reward'], True

        # Boundary failure
        if curr_dist > dcfg['boundary_radius']:
            self.get_logger().warn('Boundary violated')
            return r - 50.0, True

        return r, False

    def publish_velocity(self, vel):
        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])

        self.vel_pub.publish(msg)

    def stop(self):
        self.publish_velocity(np.zeros(3))
