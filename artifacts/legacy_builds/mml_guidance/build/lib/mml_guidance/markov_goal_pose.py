#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from typing import List, Tuple


class MarkovChain(Node):
    def __init__(self):
        super().__init__("markov_goal_pose")

        self.is_time_mode = (
            False  # Change this to False if you want to use distance mode
        )

        self.tolerance_radius = 0.15  # meters

        self.init_time = self.get_clock().now().nanoseconds / 1e9
        self.prev_goal_in = 5  # Start at the sixth state (index 5)
        self.prev_mult = 0
        self.position = np.array([0, 0])
        self.position_received = False

        # ROS parameters
        self.declare_parameter("is_sim", False)
        self.is_sim = self.get_parameter("is_sim").get_parameter_value().bool_value
        self.get_logger().info(
            f"Initializing markov_goal_pose node with parameter is_sim: {self.is_sim}"
        )

        # QoS profile for better performance
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # ← CHANGE from BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  
        )

        if self.is_sim:
            self.pose_pub = self.create_publisher(Pose, "goal_pose", 2)
            self.p = Pose()
            self.turtle_odom_sub = self.create_subscription(
                Odometry, "/robot0/odom", self.odom_cb, qos_profile
            )
        else:
            self.pose_pub = self.create_publisher(PoseStamped, "/leo/goal_pose", 2)  # ADD NAMESPACE!
            self.p = PoseStamped()
            self.turtle_odom_sub = self.create_subscription(
                msg_type=PoseStamped,
                topic="/mocap/leo/pose",
                callback=self.odom_cb,
                qos_profile=qos_profile,
            )
            self.get_logger().info("Subscribing to /mocap/leo/pose")
            self.get_logger().info("Publishing to /leo/goal_pose")


        self.road_network_V2()
        # self.goal_pose_square()
        node_freq = 10  # Hz
        self.timer = self.create_timer(1 / node_freq, self.pub_goal_pose)  # 10 Hz

    def goal_pose_square(self):
        """Generates a square of sides 2*k"""
        self.goal_list = []

        k = 0.8  # Multiplier  change this to make square bigger or smaller
        x_offset = -1.25  # change this to not crash to the net
        y_offset = 0.2
        node_positions = [
            (x_offset + 0 * k, y_offset + 0 * k, 0),  #
            (x_offset + 0 * k, y_offset + -1 * k, 270),  #
            (x_offset + 2 * k, y_offset + -1 * k, 0),  #
            (x_offset + 2 * k, y_offset + 1 * k, 90),  #
            (x_offset + 1 * k, y_offset + 2 * k, 180),  #
            (x_offset + 0 * k, y_offset + 1 * k, 180),  #
        ]
        self.node_positions_to_goal_list(node_positions)

        # transition matrix:
        # prob of going from state i to state j
        # in the goal_list states where i is the
        # row and j is the column
        self.trans_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                # for connected graph
                # [0.2, 0.0, 0.2, 0.2, 0.2, 0.2],
                # [0.2, 0.2, 0.0, 0.2, 0.2, 0.2],
                # [0.2, 0.2, 0.2, 0.0, 0.2, 0.2],
                # [0.2, 0.2, 0.2, 0.2, 0.0, 0.2],
                # [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
            ]
        )

    def road_network_V2(self):
        """Generates a road network with 8 nodes"""
        # node positions in m and orientation in deg
        node_positions = [
            (-1.5, 1, 180),  # Node 0
            (0, 1, 180),  # Node 1
            (2.1, 1, 90),  # Node 2
            (-1.5, -1, 270),  # Node 3
            (0, -1, 90),  # Node 4
            (0.3, 0, 0),  # Node 5
            (1.7, 0, 0),  # Node 6
            (2.1, -1, 270),  # Node 7
        ]
        self.node_positions_to_goal_list(node_positions)

        # transition matrix:
        self.trans_matrix = np.array(
            [
                # 0    1    2    3    4    5    6    7
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Node 0
                [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # Node 1
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 2
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 3
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Node 4
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Node 5
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 6
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Node 7
            ]
        )

    def node_positions_to_goal_list(
        self, node_positions: List[Tuple[float, float, float]]
    ):
        self.goal_list = []
        for node, pos in enumerate(node_positions):
            self.goal_list.append(
                {
                    "curr_goal": node,
                    "x": pos[0],
                    "y": pos[1],
                    "z": 0,
                    "qx": 0,
                    "qy": 0,
                    "qz": np.sin(np.deg2rad(pos[2] / 2)),
                    "qw": np.cos(np.deg2rad(pos[2] / 2)),
                }
            )
        self.n_states = len(self.goal_list)

    def odom_cb(self, msg):
        if self.is_sim:
            self.position = np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y]
            )
        else:
            self.position = np.array([
                -msg.pose.position.x,  # ← Negate x
                -msg.pose.position.y   # ← Negate y
            ])
        
        if not self.position_received:
            self.position_received = True
            self.get_logger().info(
                f"✓ First position received: [{self.position[0]:.2f}, {self.position[1]:.2f}]m"
            )

    def pub_goal_pose(self):
        """Publishes a goal pose after the goal is reached within tolerance_radius"""
        curr_goal_in = np.copy(self.prev_goal_in)
        curr_goal_pose = self.goal_list[self.prev_goal_in]
        if not self.position_received:
            self.get_logger().warn(
                "Waiting for position from /mocap/leo/pose...",
                throttle_duration_sec=2.0
            )
            return
        
        if self.is_time_mode:
            time_step = 10  # amount of seconds until next goal pose change if desired
            now = self.get_clock().now().nanoseconds / 1e9 - self.init_time
            mult = np.floor(now / time_step)
            # change goal pose if time is greater than time_step
            change = True if mult > self.prev_mult else False

            if now > 0 and now < time_step:
                curr_goal_in = 0  # Start at the first state
            elif change:
                curr_goal_in = np.random.choice(
                    np.arange(self.n_states), p=self.trans_matrix[curr_goal_in, :]
                )

            self.prev_mult = mult
        else:
            dist_to_goal = np.linalg.norm(
                self.position - np.array([curr_goal_pose["x"], curr_goal_pose["y"]])
            )
            #  print("dist to goal: ", dist_to_goal)
            if dist_to_goal < self.tolerance_radius:
                curr_goal_in = np.random.choice(
                    np.arange(self.n_states), p=self.trans_matrix[self.prev_goal_in, :]
                )

        goal_pose = self.goal_list[curr_goal_in]
        if curr_goal_in != self.prev_goal_in:
            self.get_logger().warn(
                f"New goal pose: x={goal_pose['x']:.2f}, y={goal_pose['y']:.2f} with index {curr_goal_in}"
            )

        # Restart previous values
        self.prev_goal_in = np.copy(curr_goal_in)
        # Publish the goal pose
        self.create_pose_msg_and_publish(goal_pose)

    def create_pose_msg_and_publish(self, goal_pose):
        """Creates a pose message and publishes it"""
        # Convert NumPy types to Python float
        self.p.pose.position.x = float(goal_pose["x"])
        self.p.pose.position.y = float(goal_pose["y"])
        self.p.pose.position.z = float(goal_pose["z"])
        self.p.pose.orientation.x = float(goal_pose["qx"])
        self.p.pose.orientation.y = float(goal_pose["qy"])
        self.p.pose.orientation.z = float(goal_pose["qz"])
        self.p.pose.orientation.w = float(goal_pose["qw"])
        
        # Only PoseStamped needs header (not Pose)
        if not self.is_sim:
            self.p.header.stamp = self.get_clock().now().to_msg()
            self.p.header.frame_id = "map"
        
        self.pose_pub.publish(self.p)


def main(args=None):
    rclpy.init(args=args)
    markov_chain = MarkovChain()
    rclpy.spin(markov_chain)
    markov_chain.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
