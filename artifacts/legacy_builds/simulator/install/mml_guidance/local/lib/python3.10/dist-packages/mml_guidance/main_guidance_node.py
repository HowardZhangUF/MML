#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
import time
import os
from geometry_msgs.msg import PointStamped, PoseStamped
from mml_guidance_msgs.msg import Particle, ParticleMean, ParticleArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Float32MultiArray

from .particle_filter_ros_wrapper import ParticleFilterROS2
from .guidance import Guidance


class GuidanceROS2(Node):
    def __init__(self):
        super().__init__("guidance")

        # Get ROS2 parameters
        self.is_sim = (
            self.declare_parameter("is_sim", False).get_parameter_value().bool_value
        )
        self.is_viz = (
            self.declare_parameter("is_viz", False).get_parameter_value().bool_value
        )
        guidance_mode = (
            self.declare_parameter("guidance_mode", "Information")
            .get_parameter_value()
            .string_value
        )
        prediction_method = (
            self.declare_parameter("prediction_method", "Transformer")
            .get_parameter_value()
            .string_value
        )
        N = (
            self.declare_parameter("num_particles", 500)
            .get_parameter_value()
            .integer_value
        )

        particle_filter = ParticleFilterROS2(
            N=N, prediction_method=prediction_method, is_sim=self.is_sim
        )
        self.guidance_core = Guidance(
            guidance_mode=guidance_mode,
            prediction_method=prediction_method,
            filter=particle_filter,
        )

        # Get additional ROS2 parameters
        self.declare_parameter("predict_window", self.guidance_core.K)
        self.declare_parameter("num_sampled_particles", self.guidance_core.N_s)

        # Occlusion parameters
        occlusion_widths = [1, 1]  # Default widths
        occlusion_centers = [-1.25, -0.6, 0.35, 0.2]  # Default centers
        self.declare_parameter(
            "occlusion_widths", occlusion_widths
        ).get_parameter_value().integer_array_value
        self.declare_parameter(
            "occlusion_centers", occlusion_centers
        ).get_parameter_value().double_array_value

        # ROS2 Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 1
        )
        self.err_tracking_pub = self.create_publisher(PointStamped, "err_tracking", 1)

        # ROS2 Subscribers
        if self.is_sim:
            self.turtle_odom_sub = self.create_subscription(
                Odometry, "/robot0/odom", self.turtle_odom_cb, 1
            )
            self.quad_odom_sub = self.create_subscription(
                PoseStamped, "/pose_stamped", self.quad_odom_cb, 1
            )
        else:
            self.quad_odom_sub = self.create_subscription(
                PoseStamped, "/quad_pose_stamped", self.quad_odom_cb, 1
            )
            self.turtle_odom_sub = self.create_subscription(
                PoseStamped, "/turtle_pose_stamped", self.turtle_odom_cb, 1
            )
            self.turtle_odom_sub = self.create_subscription(
                Odometry, "/odom", self.turtle_hardware_odom_cb, 1
            )

        # Visualization publishers
        if self.is_viz:
            self.particle_pub = self.create_publisher(ParticleMean, "xyTh_estimate", 1)
            self.particle_pred_pub = self.create_publisher(
                ParticleArray, "xyTh_predictions", 1
            )
            self.sampled_index_pub = self.create_publisher(
                Float32MultiArray, "sampled_index", 1
            )
            self.err_estimation_pub = self.create_publisher(
                PointStamped, "err_estimation", 1
            )
            self.meas_pub = self.create_publisher(PointStamped, "noisy_measurement", 1)
            self.entropy_pub = self.create_publisher(Float32, "entropy", 1)
            self.info_gain_pub = self.create_publisher(Float32, "info_gain", 1)
            self.eer_time_pub = self.create_publisher(Float32, "eer_time", 1)
            self.det_cov_pub = self.create_publisher(
                Float32, "xyTh_estimate_cov_det", 1
            )
            self.n_eff_pub = self.create_publisher(Float32, "n_eff_particles", 1)
            self.update_pub = self.create_publisher(Bool, "is_update", 1)
            self.occ_pub = self.create_publisher(Bool, "is_occlusion", 1)
            self.fov_pub = self.create_publisher(Float32MultiArray, "fov_coord", 1)
            self.des_fov_pub = self.create_publisher(
                Float32MultiArray, "des_fov_coord", 1
            )

        self.get_logger().info(
            f"Initializing guidance node with parameter is_sim: {self.is_sim}"
        )
        self.get_logger().info(f"...and parameter is_viz: {self.is_viz}")
        self.get_logger().info(
            f"Quadcopter in guidance mode: {self.guidance_core.guidance_mode}"
        )
        self.get_logger().info(
            f"... and in prediction method: {self.guidance_core.prediction_method}"
        )
        self.get_logger().info(
            "Number of particles for the Bayes Filter: %d" % self.guidance_core.N
        )

    def turtle_odom_cb(self, msg):
        """Callback for the turtlebot odometry"""
        if self.guidance_core.init_finished:
            if self.is_sim:
                position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
                orientation = [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
                linear_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
                angular_vel = [msg.twist.twist.angular.z]
            else:
                position = [msg.pose.position.x, msg.pose.position.y]
                orientation = [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
                linear_vel = None
                angular_vel = None

            self.guidance_core.update_target_pose(
                position, orientation, linear_vel, angular_vel
            )

    def turtle_hardware_odom_cb(self, msg):
        """Callback for hardware turtlebot velocities"""
        if self.guidance_core.init_finished:
            linear_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
            angular_vel = [msg.twist.twist.angular.z]
            self.guidance_core.linear_velocity = np.array(linear_vel)
            self.guidance_core.angular_velocity = np.array(angular_vel)

    def quad_odom_cb(self, msg):
        """Callback for quadcopter position"""
        if self.guidance_core.init_finished:
            if self.is_sim:
                position = [msg.pose.position.x, -msg.pose.position.y]
            else:
                position = [msg.pose.position.x, msg.pose.position.y]
            self.guidance_core.update_agent_position(position)

    def pub_desired_state(self):
        """Publishes messages related to desired state"""
        if self.guidance_core.init_finished:
            self.guidance_core.update_goal_position()
            ds = PoseStamped()

            if self.guidance_core.position_following or self.is_sim:
                ds.pose.position.x = self.guidance_core.goal_position[0]
                ds.pose.position.y = self.guidance_core.goal_position[1]
            else:
                ds.pose.position.x = 0
                ds.pose.position.y = 0

            ds.pose.position.z = self.guidance_core.height

            # Clip to lab boundaries
            ds.pose.position.x = np.clip(
                ds.pose.position.x,
                self.guidance_core.filter.APRILab_dims[0][0],
                self.guidance_core.filter.APRILab_dims[1][0],
            )
            ds.pose.position.y = np.clip(
                ds.pose.position.y,
                self.guidance_core.filter.APRILab_dims[0][1],
                self.guidance_core.filter.APRILab_dims[1][1],
            )

            self.pose_pub.publish(ds)

            # Tracking error
            FOV_err = (
                self.guidance_core.quad_position
                - self.guidance_core.actual_turtle_pose[:2]
            )
            err_tracking_msg = PointStamped()
            err_tracking_msg.point.x = FOV_err[0]
            err_tracking_msg.point.y = FOV_err[1]
            self.err_tracking_pub.publish(err_tracking_msg)

            if self.is_viz:
                self._publish_visualization_data(ds)

    def _publish_visualization_data(self, ds):
        """Publish visualization data"""
        # FOV pub
        fov_msg = Float32MultiArray()
        fov_matrix = np.array(
            [
                [self.guidance_core.FOV[0], self.guidance_core.FOV[2]],
                [self.guidance_core.FOV[0], self.guidance_core.FOV[3]],
                [self.guidance_core.FOV[1], self.guidance_core.FOV[3]],
                [self.guidance_core.FOV[1], self.guidance_core.FOV[2]],
                [self.guidance_core.FOV[0], self.guidance_core.FOV[2]],
            ]
        )
        fov_msg.data = fov_matrix.flatten("C").tolist()
        self.fov_pub.publish(fov_msg)

        # Desired FOV
        des_fov = self.guidance_core.construct_FOV(
            np.array([ds.pose.position.x, ds.pose.position.y])
        )
        des_fov_matrix = np.array(
            [
                [des_fov[0], des_fov[2]],
                [des_fov[0], des_fov[3]],
                [des_fov[1], des_fov[3]],
                [des_fov[1], des_fov[2]],
                [des_fov[0], des_fov[2]],
            ]
        )
        fov_msg.data = des_fov_matrix.flatten("C").tolist()
        self.des_fov_pub.publish(fov_msg)

        # Update and measurement status
        update_msg = Bool()
        update_msg.data = self.guidance_core.filter.is_update
        self.update_pub.publish(update_msg)

        if self.guidance_core.filter.is_update:
            meas_msg = PointStamped()
            meas_msg.point.x = self.guidance_core.noisy_turtle_pose[0]
            meas_msg.point.y = self.guidance_core.noisy_turtle_pose[1]
            self.meas_pub.publish(meas_msg)

        occ_msg = Bool()
        occ_msg.data = self.guidance_core.filter.is_occlusion
        self.occ_pub.publish(occ_msg)

    def pub_pf(self):
        """Publishes the particles and the mean of the particle filter"""
        if self.guidance_core.init_finished:
            mean_msg = ParticleMean()
            mean_msg.mean.x = self.guidance_core.filter.weighted_mean[0]
            mean_msg.mean.y = self.guidance_core.filter.weighted_mean[1]
            mean_msg.mean.yaw = np.linalg.norm(
                self.guidance_core.filter.weighted_mean[2:4]
            )

            for ii in range(self.guidance_core.N):
                particle_msg = Particle()
                particle_msg.x = self.guidance_core.filter.particles[-1, ii, 0]
                particle_msg.y = self.guidance_core.filter.particles[-1, ii, 1]
                particle_msg.yaw = np.linalg.norm(
                    self.guidance_core.filter.particles[-1, ii, 2:4]
                )
                particle_msg.weight = self.guidance_core.filter.weights[ii]
                mean_msg.all_particle.append(particle_msg)

            mean_msg.cov = np.diag(self.guidance_core.filter.var).flatten("C").tolist()
            self.particle_pub.publish(mean_msg)

            # Error estimation
            err_msg = PointStamped()
            err_msg.point.x = (
                self.guidance_core.filter.weighted_mean[0]
                - self.guidance_core.actual_turtle_pose[0]
            )
            err_msg.point.y = (
                self.guidance_core.filter.weighted_mean[1]
                - self.guidance_core.actual_turtle_pose[1]
            )
            self.err_estimation_pub.publish(err_msg)

            # Various metrics
            neff_msg = Float32()
            neff_msg.data = self.guidance_core.filter.neff
            self.n_eff_pub.publish(neff_msg)

            info_gain_msg = Float32()
            info_gain_msg.data = float(self.guidance_core.EER_range[0])
            self.entropy_pub.publish(info_gain_msg)

            eer_time_msg = Float32()
            eer_time_msg.data = float(self.t_EER)  # TODO
            self.eer_time_pub.publish(eer_time_msg)

            entropy_msg = Float32()
            entropy_msg.data = float(self.guidance_core.Hp_t)
            self.entropy_pub.publish(entropy_msg)

            if self.guidance_core.prediction_method == "KF":
                det_msg = Float32()
                det_msg.data = np.linalg.det(self.guidance_core.kf.P[:2, :2])
                self.det_cov_pub.publish(det_msg)

    def information_driven_guidance(self, event=None):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the position of the particle
        that maximizes the EER.
        Output:
        eer_particle: the index of the particle that maximizes the EER which
        we propagate to choose the goal position
        """
        future_parts = np.copy(self.guidance_core.sampled_particles)
        for k in range(self.guidance_core.K):  # propagate k steps in the future
            pred_msg = ParticleArray()
            future_parts = self.guidance_core.propagate_particles(
                future_parts,
                self.guidance_core.K,
                self.guidance_core.N_s,
            )
            if self.is_viz:
                assert future_parts.shape == (
                    self.filter.N_th,
                    self.N_s,
                    self.filter.Nx,
                )
                mean_msg = ParticleMean()
                for ii in range(self.N_s):
                    particle_msg = Particle()
                    particle_msg.x = future_parts[-1, ii, 0]
                    particle_msg.y = future_parts[-1, ii, 1]
                    particle_msg.weight = self.filter.weights[ii]
                    mean_msg.all_particle.append(particle_msg)
                pred_msg.particle_array.append(mean_msg)

        self.guidance_core.information_driven_guidance(future_parts)

        if self.is_viz:
            if len(pred_msg.particle_array) > self.K:
                pred_msg.particle_array.pop(0)  # eliminate the oldest particle
            self.particle_pred_pub.publish(pred_msg)  # publish the history
            # publish the sampled index array
            # Convert numpy array to Float32MultiArray before publishing
            float32_multi_array_msg = Float32MultiArray()
            float32_multi_array_msg.data = (
                self.sampled_index.astype(float).flatten().tolist()
            )
            self.sampled_index_pub.publish(float32_multi_array_msg)

    def guidance_pf(self):
        """Runs the particle filter loop based on the estimation method"""
        self.guidance_core.guidance_pf(t=self.get_clock().now().nanoseconds / 1e9)

    def schedule_shutdown(self, time_to_shutdown):
        """Schedule node shutdown"""

        def shutdown():
            self.get_logger().fatal(
                "Timer expired or user terminated. Stopping the node..."
            )
            time.sleep(0.1)
            os.system("ros2 lifecycle set /drone_guidance shutdown")
            os.system("ros2 lifecycle set /mml_pf_visualization shutdown")

        self.shutdown_timer = self.create_timer(time_to_shutdown, shutdown)


def main(args=None):
    rclpy.init(args=args)
    guidance = Guidance()

    time_to_shutdown = 1000  # in seconds
    guidance.schedule_shutdown(time_to_shutdown=time_to_shutdown)

    # Running functions at a certain rate
    guidance.create_timer(1.0 / 3.0, guidance.guidance_pf)
    guidance.create_timer(1.0 / 3.0, guidance.guidance_core.current_entropy)

    if guidance.guidance_core.guidance_mode == "Information":
        guidance.create_timer(1.0 / 2.5, guidance.information_driven_guidance)

    # Publish topics
    guidance.create_timer(1.0 / 3.0, guidance.pub_desired_state)

    if guidance.is_viz:
        guidance.create_timer(1.0 / 3.0, guidance.pub_pf)

    try:
        rclpy.spin(guidance)
    except KeyboardInterrupt:
        guidance.guidance_core.shutdown()
    finally:
        guidance.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
