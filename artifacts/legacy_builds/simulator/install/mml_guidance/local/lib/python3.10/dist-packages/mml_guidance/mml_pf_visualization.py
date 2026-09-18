#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from mml_guidance_msgs.msg import Particle, ParticleMean, ParticleArray
from std_msgs.msg import Bool, Float32, Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import matplotlib.gridspec as gridspec

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

# extend the display of the visualization in some part of the screen
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        f.canvas.manager.window.move(x, y)
    f.set_size_inches(19, 24.0)

class MMLPFVisualization(Node):
    def __init__(self):
        super().__init__('mml_pf_visualization')

        self.initialization_finished = False
        self.is_sim = self.declare_parameter("is_sim", False).get_parameter_value().bool_value
        self.declare_parameter("occlusion_widths", [0,0])
        self.declare_parameter("occlusion_centers", [0,0,0,0])

        self.data_size = 200
        self.update_size = 6

        # ROS2 equivalent to rospy.Rate
        self.loop_rate = self.create_rate(100)

        plt.ion()  # Interactive mode

        # Creating subscribers
        self.mml_pf_subscriber = self.create_subscription(
            ParticleMean, "xyTh_estimate", self.mml_pf_callback, 10
        )
        self.sampled_index_subscriber = self.create_subscription(
            Float32MultiArray, "sampled_index", self.si_cb, 10
        )
        self.mml_pred_pf_subscriber = self.create_subscription(
            ParticleArray, "xyTh_predictions", self.mml_pred_pf_callback, 10
        )
        self.mml_errEstimate_subscriber = self.create_subscription(
            PointStamped, "err_estimation", self.mml_err_callback, 10
        )
        self.mml_entropy_subscriber = self.create_subscription(
            Float32, "entropy", self.entropy_callback, 10
        )
        if self.is_sim:
            self.true_position_subscriber = self.create_subscription(
                Odometry, "odom", self.odom_callback, 10
            )
            self.mocap_msg = Odometry()
        else:
            self.true_position_subscriber = self.create_subscription(
                PoseStamped, "odom", self.odom_callback, 10
            )
            self.mocap_msg = PoseStamped()
        self.noisy_measurements_subscriber = self.create_subscription(
            PointStamped, "/noisy_measurement", self.measurement_callback, 10
        )
        self.update_subscriber = self.create_subscription(
            Bool, "is_update", self.upd_callback, 10
        )
        self.fov_subscriber = self.create_subscription(
            Float32MultiArray, "fov_coord", self.fov_callback, 10
        )
        self.des_fov_subscriber = self.create_subscription(
            Float32MultiArray, "des_fov_coord", self.des_fov_callback, 10
        )

        # Initialize matplotlib figure UI
        self.fig = plt.figure()
        self.fig2 = plt.figure()
        gs = gridspec.GridSpec(ncols=3, nrows=3)
        move_figure(self.fig, 0, 0)

        self.fig_ax1 = self.fig.add_subplot(gs[:, :])
        self.fig_ax1.set_title("Map")
        self.fig_ax2 = self.fig2.add_subplot(gs[0, 1])
        self.fig_ax2.set_title("Est x vs Real x")
        self.fig_ax3 = self.fig2.add_subplot(gs[1, 1])
        self.fig_ax3.set_title("Est y vs Real y")
        self.fig_ax4 = self.fig2.add_subplot(gs[2, 1])
        self.fig_ax4.set_title("Est yaw vs Real yaw")
        self.fig_ax5 = self.fig2.add_subplot(gs[:, 2])
        self.fig_ax5.set_title("Entropy")
        plt.close(self.fig2)

        # set the map image boundaries and resize it to respect aspect ratio
        self.limits = {"x": [-2.5, 3.0], "y": [-1.6, 1.6]}

        self.xErrList = []
        self.yErrList = []
        self.yawErrList = []
        self.timestamplist = []

        self.entropyList = []
        self.xReallist = []
        self.yReallist = []
        self.yawReallist = []
        self.timestampReallist = []
        self.update_msg = Bool()
        self.measurement_msg = None
        self.update_msg.data = True
        self.update_t = np.zeros(self.update_size)
        self.goal_pose = np.array([0, 0])
        self.fov = np.zeros((5, 2))
        self.des_fov = np.zeros((5, 2))
        self.particles = np.zeros((1, 2))
        self.K = self.declare_parameter("predict_window", 4).get_parameter_value().integer_value
        self.N_s = self.declare_parameter("num_sampled_particles", 25).get_parameter_value().integer_value
        self.future_parts = np.zeros((self.K, self.N_s, 2))
        self.plot_prediction = False
        self.sampled_index = np.arange(self.N_s)

        self.x_sigmaList = []
        self.y_sigmaList = []
        self.yaw_sigmaList = []

        # Road Network
        #self.road_network = self.declare_parameter("road_network", None).get_parameter_value().array_value
        self.road_network = None
        if self.road_network is None:
            self.road_network = np.array(
                [
                    (1.7, 0),  # 6
                    (0.3, 0),  # 5
                    (0, -1),  # 4
                    (2, -1),  # 7
                    (1.7, 0),  # 6
                    (2, 1),  # 2
                    (0, 1),  # 1
                    (-1.5, -1),  # 3
                    (-1.5, 1),  # 0
                    (0, 1),  # 1
                    (-1.5, 1),  # 0
                    (0, -1),  # 4
                ]
            )

        self.plot_flag = False
        self.initialization_finished = True

    def is_inside_limits(self, position):
        result = (
            position[0] > self.limits["x"][0]
            and position[0] < self.limits["x"][1]
            and position[1] > self.limits["y"][0]
            and position[1] < self.limits["y"][1]
        )
        # print("is_inside_limits: ", result) if not result else None
        return result

    def joy_callback(self, msg):
        # Not used in ROS2
        pass

    def upd_callback(self, msg):
        if self.initialization_finished:
            if msg.data:
                t = self.get_clock().now().to_msg().sec
                self.update_t[:-1] = self.update_t[1:]
                self.update_t[-1] = t
            self.update_msg = msg

    def fov_callback(self, msg):
        if self.initialization_finished:
            # [bottom left, top left, top right, bottom right, bottom left]
            self.fov = np.array(msg.data).reshape((5, 2))

    def des_fov_callback(self, msg):
        if self.initialization_finished:
            # [bottom left, top left, top right, bottom right, bottom left]
            self.des_fov = np.array(msg.data).reshape((5, 2))

    def mml_pf_callback(self, msg):
        if self.initialization_finished:
            self.particle_mean = np.array([msg.mean.x, msg.mean.y])
            self.cov = np.array(msg.cov)

            # setup the particle array
            particle_list = [[data.x, data.y] for data in msg.all_particle]
            self.particles = np.array(particle_list)

            # compute sigma values
            x_sigma = 3 * math.sqrt(msg.cov[0])
            y_sigma = 3 * math.sqrt(msg.cov[4])
            yaw_sigma = 3 * math.sqrt(msg.cov[8])
            self.x_sigmaList.append(x_sigma)
            self.y_sigmaList.append(y_sigma)
            self.yaw_sigmaList.append(yaw_sigma)

            self.plot_flag = True

    def mml_pred_pf_callback(self, msg):
        if self.initialization_finished == True:
            self.plot_prediction = True
            for k in range(self.K):
                particle_list = []
                for data in msg.particle_array[k].all_particle:
                    particle_list.append([data.x, data.y])
                particles = np.array(particle_list)
                self.future_parts[k, :, :] = particles

            self.plot_flag = True

    def si_cb(self, msg):
        if self.initialization_finished == True:
            self.sampled_index = np.array(msg.data).astype(int)

    def mml_err_callback(self, msg):
        if self.initialization_finished == True:
            # estimate error holder update
            self.xErrList.append(msg.point.x)
            self.yErrList.append(msg.point.y)

            # change yaw boundaries to -pi to pi
            errYaw = msg.point.z % (2 * math.pi)
            if errYaw > math.pi:
                errYaw = errYaw - 2 * math.pi
            if errYaw < -math.pi:
                errYaw = errYaw + 2 * math.pi
            # error on yaw update
            self.yawErrList.append(errYaw)
            # according timestamp update
            self.timestamplist.append(self.get_clock().now().seconds_nanoseconds()[0] )
            self.plot_flag = True

    def entropy_callback(self, msg):
        if self.initialization_finished == True:
            self.entropyList.append(msg.data)
            self.plot_flag = True

    def odom_callback(self, msg):
        if self.initialization_finished == True:
            # mocap holder update
            self.mocap_msg = msg

    def measurement_callback(self, msg):
        if self.initialization_finished == True:
            # mocap holder update
            self.measurement_msg = msg

    def getCircle(self):
        """generate a python list representing a normalized circle"""
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.block([[np.cos(theta)], [np.sin(theta)]])

        return circle

    def plotCov(self, ax, avg, cov):
        """plot the covariance ellipse on top of the particles"""
        # compute eigen values
        y, v = np.linalg.eig(cov)
        # take the real part of the eigen value
        y, v = np.real(y), np.real(v)
        # get all the 3 sigmas values
        r = np.sqrt(7.814 * np.abs(y))  # 5.991 for 95% confidence. 7.814 for 3dof

        # generate a normalized circle
        circle = self.getCircle()
        # compute the ellipse shape
        ellipse = np.matmul(v, (r[:, None] * circle)) + avg[:2, None]
        # pot the ellipse
        self.fig_ax1.plot(ellipse[0], ellipse[1], color="g")

    # def adding_attrited_value_to_plot(self):

    def visualization_spin(self):

        if self.initialization_finished:

            if self.plot_flag:
                # clear and reinitialize all the plots
                self.fig_ax1.clear()
                self.fig_ax2.clear()
                self.fig_ax3.clear()
                self.fig_ax4.clear()
                self.fig_ax5.clear()
                self.fig_ax1.set_title("Map")
                self.fig_ax2.set_title("Est x vs Real x")
                self.fig_ax3.set_title("Est y vs Real y")
                self.fig_ax4.set_title("Est yaw vs Real yaw")
                self.fig_ax5.set_title("Entropy")
                # self.fig_ax1.set_xlim(self.limits["x"])
                # self.fig_ax1.set_ylim(self.limits["y"])

                # plot the road network
                self.fig_ax1.plot(
                    self.road_network[:, 0],
                    self.road_network[:, 1],
                    marker=".",
                    markersize=5,
                    alpha=0.6,
                    color="k",
                    label="Road Network",
                )

                # plot the particles
                # TODO: change from only sampled to all particles
                self.fig_ax1.scatter(
                    self.particles[self.sampled_index, 0],
                    self.particles[self.sampled_index, 1],
                    marker=".",
                    color="k",
                    label="Sampled !!! Particles ",
                    clip_on=True,
                )
                # plot the current estimate position
                self.fig_ax1.plot(
                    self.particle_mean[0],
                    self.particle_mean[1],
                    marker="P",
                    markersize=10.0,
                    color="g",
                    label="Estimated position ",
                    clip_on=True,
                )

                # plot spaguetti plots and scatter using the future particles and the sampled index
                # TODO: change from blue to black
                if self.plot_prediction:
                    slope = (0.5 - 0.05) / (self.K - 1)
                    for k in range(self.K):
                        self.fig_ax1.scatter(
                            self.future_parts[k, :, 0],
                            self.future_parts[k, :, 1],
                            marker=".",
                            color="b",
                            alpha=0.5 - slope * k,
                            clip_on=True,
                        )
                        counter = 0
                        for ii in self.sampled_index:
                            if k == 0:
                                self.fig_ax1.plot(
                                    [
                                        self.particles[ii, 0],
                                        self.future_parts[k, counter, 0],
                                    ],
                                    [
                                        self.particles[ii, 1],
                                        self.future_parts[k, counter, 1],
                                    ],
                                    color="b",
                                    alpha=0.2,
                                    clip_on=True,
                                )
                            else:
                                self.fig_ax1.plot(
                                    [
                                        self.future_parts[k - 1, counter, 0],
                                        self.future_parts[k, counter, 0],
                                    ],
                                    [
                                        self.future_parts[k - 1, counter, 1],
                                        self.future_parts[k, counter, 1],
                                    ],
                                    color="b",
                                    alpha=0.2,
                                    clip_on=True,
                                )
                            counter += 1
                    self.fig_ax1.scatter(
                        self.fov[0, 0],
                        self.fov[0, 1],
                        marker=".",
                        color="b",
                        alpha=0.01,
                        label="Propagated Particles",
                        clip_on=True,
                    )  # fake for legend

                # plot the real position
                if self.is_sim:
                    self.fig_ax1.plot(
                        self.mocap_msg.pose.pose.position.x,
                        self.mocap_msg.pose.pose.position.y,
                        marker="P",
                        markersize=10.0,
                        color="m",
                        label="True position ",
                        clip_on=True,
                    )
                else:
                    self.fig_ax1.plot(
                        self.mocap_msg.pose.position.x,
                        self.mocap_msg.pose.position.y,
                        marker="P",
                        markersize=10.0,
                        color="m",
                        label="True position ",
                        clip_on=True,
                    )
                # plot the desired fov
                self.fig_ax1.plot(
                    self.des_fov[:, 0],
                    self.des_fov[:, 1],
                    marker=".",
                    markersize=1.0,
                    color="b",
                    label="Action FOV",
                    clip_on=True,
                )
                # [bottom left, top left, top right, bottom right, bottom left]
                act_x = (self.des_fov[2, 0] - self.des_fov[0, 0]) / 2.0 + self.des_fov[
                    0, 0
                ]
                act_y = (self.des_fov[1, 1] - self.des_fov[0, 1]) / 2.0 + self.des_fov[
                    0, 1
                ]
                self.fig_ax1.scatter(
                    act_x,
                    act_y,
                    marker="+",
                    color="b",
                    label="Action chosen",
                    clip_on=True,
                )
                # plot the fov
                self.fig_ax1.plot(
                    self.fov[:, 0],
                    self.fov[:, 1],
                    marker=".",
                    markersize=1.0,
                    color="r",
                    label="Field of View ",
                    clip_on=True,
                )
                quad_x = (self.fov[2, 0] - self.fov[0, 0]) / 2.0 + self.fov[0, 0]
                quad_y = (self.fov[1, 1] - self.fov[0, 1]) / 2.0 + self.fov[0, 1]
                self.fig_ax1.scatter(
                    quad_x,
                    quad_y,
                    marker="+",
                    color="r",
                    label="Quad position ",
                    clip_on=True,
                )
                # plot the occlusion
                occ_centers = list(self.get_parameter('occlusion_centers').get_parameter_value().double_array_value)
                occ_widths = list(self.get_parameter('occlusion_widths').get_parameter_value().double_array_value)
                occ_widths = [1, 1]
                occ_centers = [-1.25, -0.6, 0.35, 0.2]  # flattened [x1, y1, x2, y2] to send as parameters
                occ_centers = [occ_centers[:2], occ_centers[2:]]
                # TODO: make parameters
                if occ_centers != None:
                    for occ_center, width in zip(occ_centers, occ_widths):
                        x_dim = np.array([-width, width, width, -width, -width])
                        y_dim = np.array([-width, -width, width, width, -width])
                        self.fig_ax1.plot(
                            occ_center[0] + x_dim / 2,
                            occ_center[1] + y_dim / 2,
                            marker=".",
                            markersize=0.5,
                            color="k",
                        )
                    self.fig_ax1.plot(
                        occ_center[0] + x_dim / 2,
                        occ_center[1] + y_dim / 2,
                        marker=".",
                        markersize=0.5,
                        color="k",
                        label="Occlusion ",
                    )
                # self.fig_ax1.plot([], [], marker=">",
                #                   markersize=10., color="black", label="Measurement")
                # add legend
                self.fig_ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

                # shows the camera picture to indicate measurement
                # if self.update_msg.data:
                #    self.fig_ax1.imshow(self.img_cam, extent=[2.5, 4.5, -3.4, -1.8])

                # plot estimated arrow theta arrow on the map
                # if the update happen 0.25 sec before then use a yellow arrow
                # plot real arrow theta arrow on the map
                if self.is_sim:
                    real_theta = euler_from_quaternion(
                        self.mocap_msg.pose.pose.orientation.x,
                        self.mocap_msg.pose.pose.orientation.y,
                        self.mocap_msg.pose.pose.orientation.z,
                        self.mocap_msg.pose.pose.orientation.w,
                    )[2]
                    # self.fig_ax1.arrow(
                    #     self.mocap_msg.pose.pose.position.x,
                    #     self.mocap_msg.pose.pose.position.y,
                    #     0.25 * math.cos(real_theta),
                    #     0.25 * math.sin(real_theta),
                    #     width=0.05,
                    #     color="magenta",
                    #     label="Estimated yaw",
                    # )
                else:
                    real_theta = euler_from_quaternion(
                        self.mocap_msg.pose.orientation.x,
                        self.mocap_msg.pose.orientation.y,
                        self.mocap_msg.pose.orientation.z,
                        self.mocap_msg.pose.orientation.w,
                    )[2]
                    self.fig_ax1.arrow(
                        self.mocap_msg.pose.position.x,
                        self.mocap_msg.pose.position.y,
                        0.25 * math.cos(real_theta),
                        0.25 * math.sin(real_theta),
                        width=0.05,
                        color="magenta",
                        label="Estimated yaw",
                    )
                # plot the measurement
                if self.measurement_msg != None:
                    self.fig_ax1.plot(
                        self.measurement_msg.point.x,
                        self.measurement_msg.point.y,
                        marker="+",
                        markersize=10.0,
                        color="orange",
                        label="Noisy Measurements",
                        clip_on=True,
                    )

                # plot particles covariances
                self.plotCov(
                    self.fig_ax1,
                    self.particle_mean,
                    self.cov.reshape((3, 3))[:2, :2],
                )
                self.fig_ax1.grid(True)

                # trim extra data to keep only the most updated data plotted
                if len(self.timestamplist) > self.data_size:
                    self.xErrList = self.xErrList[-self.data_size :]
                    self.yErrList = self.yErrList[-self.data_size :]
                    self.yawErrList = self.yawErrList[-self.data_size :]
                    self.timestamplist = self.timestamplist[-self.data_size :]
                    self.x_sigmaList = self.x_sigmaList[-self.data_size :]
                    self.y_sigmaList = self.y_sigmaList[-self.data_size :]
                    self.yaw_sigmaList = self.yaw_sigmaList[-self.data_size :]
                    self.entropyList = self.entropyList[-self.data_size :]

                timestamplist = np.copy(self.timestamplist)[0 : self.data_size]
                xErrList = np.copy(self.xErrList)[0 : self.data_size]
                yErrList = np.copy(self.yErrList)[0 : self.data_size]
                yawErrList = np.copy(self.yawErrList)[0 : self.data_size]
                x_sigmaList = np.copy(self.x_sigmaList)[0 : self.data_size]
                y_sigmaList = np.copy(self.y_sigmaList)[0 : self.data_size]
                yaw_sigmaList = np.copy(self.yaw_sigmaList)[0 : self.data_size]
                entropyList = np.copy(self.entropyList)[0 : self.data_size]

                # plot the estimates errors and their sigmas
                if len(timestamplist) == len(xErrList) and len(x_sigmaList) == len(
                    timestamplist
                ):
                    self.fig_ax2.set_ylim(-1, 1)
                    self.fig_ax2.plot(timestamplist, xErrList)
                    self.fig_ax2.plot(timestamplist, (x_sigmaList), color="r")
                    self.fig_ax2.plot(timestamplist, (-x_sigmaList), color="r")
                    self.fig_ax2.grid(True)

                if len(timestamplist) == len(yErrList) and len(y_sigmaList) == len(
                    timestamplist
                ):
                    self.fig_ax3.set_ylim(-1, 1)
                    self.fig_ax3.plot(timestamplist, yErrList)
                    self.fig_ax3.plot(timestamplist, (y_sigmaList), color="r")
                    self.fig_ax3.plot(timestamplist, (-y_sigmaList), color="r")
                if len(self.timestamplist) == len(self.yawErrList) and len(
                    yaw_sigmaList
                ) == len(timestamplist):
                    self.fig_ax4.set_ylim(-0.4, 0.4)
                    self.fig_ax4.plot(timestamplist, yawErrList)
                    self.fig_ax4.plot(timestamplist, (yaw_sigmaList), color="r")
                    self.fig_ax4.plot(timestamplist, (-yaw_sigmaList), color="r")

                # if len(self.timestamplist) == len(self.entropyList) and len(
                #    entropyList
                # ) == len(timestamplist):
                # self.fig_ax5.set_ylim(0,1)
                self.fig_ax5.plot(timestamplist, entropyList[: len(timestamplist)])

                self.plot_flag = False

                # orange bars to show update times
                # for fig in [self.fig_ax2, self.fig_ax3]:
                #    for t in self.update_t:
                #        fig.plot([t,t], [-10,10], color="orange")
            self.fig_ax1.autoscale(enable=False)
            self.fig_ax1.set_xlim(left=self.limits["x"][0], right=self.limits["x"][1])
            self.fig_ax1.set_ylim(bottom=self.limits["y"][0], top=self.limits["y"][1])

            # update all the plots and display them
            plt.show()

            plt.pause(0.01)

            self.loop_rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = MMLPFVisualization()
    node.create_timer(1.0 / 60, node.visualization_spin)

    while rclpy.ok():
        #node.visualization_spin()
        rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()