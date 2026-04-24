import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import numpy as np
import subprocess
import threading


class EvaderNode(Node):
    def __init__(self):
        super().__init__('evader_node')

        self.declare_parameter('speed', 0.5)
        self.declare_parameter('start_x', 3.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 3.0)
        self.declare_parameter('direction_change_interval', 2.0)
        self.declare_parameter('use_gazebo', True)

        self.speed      = self.get_parameter('speed').value
        self.start_pos  = np.array([
            self.get_parameter('start_x').value,
            self.get_parameter('start_y').value,
            self.get_parameter('start_z').value,
        ], dtype=np.float64)
        self.pos        = self.start_pos.copy()
        self.vel        = np.zeros(3, dtype=np.float64)
        self.interval   = self.get_parameter('direction_change_interval').value
        self.use_gazebo = self.get_parameter('use_gazebo').value
        self._gz_counter = 0
        self._gz_busy = False  # prevent concurrent gz calls

        self.odom_pub = self.create_publisher(Odometry, '/evader/odom', 10)
        self.create_service(Trigger, '/evader/reset', self._reset_cb)
        self.create_timer(0.05, self._update)
        self.create_timer(self.interval, self._change_direction)

        self.get_logger().info(
            f'EvaderNode started — speed={self.speed} start={self.start_pos}'
        )
        self._change_direction()

    def _reset_cb(self, request, response):
        self.pos = self.start_pos.copy()
        self.vel = np.zeros(3)
        self._change_direction()
        self._move_gazebo_model()
        response.success = True
        response.message = f'Reset to {self.start_pos}'
        return response

    def _change_direction(self):
        if self.speed == 0.0:
            self.vel = np.zeros(3)
            return
        d = np.random.randn(3)
        d[2] = 0.0   # no vertical velocity component
        n = np.linalg.norm(d)
        if n > 0:
            d /= n
        self.vel = d * self.speed

    def _update(self):
        dt = 0.05
        self.pos += self.vel * dt

        # XY boundary
        if np.linalg.norm(self.pos[:2]) > 15.0:
            self.vel[:2] *= -1.0

        # Z — keep fixed at start height
        self.pos[2] = self.start_pos[2]
        self.vel[2] = 0.0

        self._publish_odom()

        # move Gazebo at 2 Hz (every 10 updates)
        self._gz_counter += 1
        if self.use_gazebo and self._gz_counter >= 10 and not self._gz_busy:
            self._gz_counter = 0
            threading.Thread(target=self._move_gazebo_model, daemon=True).start()

    def _publish_odom(self):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'evader'
        msg.pose.pose.position.x = self.pos[0]
        msg.pose.pose.position.y = self.pos[1]
        msg.pose.pose.position.z = self.pos[2]
        msg.twist.twist.linear.x = self.vel[0]
        msg.twist.twist.linear.y = self.vel[1]
        msg.twist.twist.linear.z = self.vel[2]
        self.odom_pub.publish(msg)

    def _move_gazebo_model(self):
        self._gz_busy = True
        x = float(self.pos[0])
        y = float(self.pos[1])
        z = float(self.pos[2])
        req = f'name: "crazyflie" position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}} orientation: {{w: 1}}'
        try:
            subprocess.run(
                ['gz', 'service', '-s', '/world/pursuit_world/set_pose',
                 '--reqtype', 'gz.msgs.Pose',
                 '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '500',
                 '--req', req],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
        except Exception:
            pass
        self._gz_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = EvaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
