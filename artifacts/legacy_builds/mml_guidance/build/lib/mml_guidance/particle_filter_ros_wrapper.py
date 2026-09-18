#!/usr/bin/env python3
import os
from rclpy.clock import Clock, ClockType
from ament_index_python.packages import get_package_share_directory
from .ParticleFilter import ParticleFilter

# ROS2 Clock for timing
clock = Clock(clock_type=ClockType.ROS_TIME)


class ParticleFilterROS2(ParticleFilter):
    def __init__(
        self, num_particles=10, prediction_method="NN", is_sim=False, drone_height=2.0
    ):
        # Get ROS2 package path for model loading
        # TODO: remove this file
        pkg_path = get_package_share_directory("mml_guidance")

        # Construct model path based on prediction method
        model_path = None
        if prediction_method in {"NN", "Transformer"}:
            is_occlusions_weights = True
            is_time_weights = True
            prefix_name = "noisy_5_v2_"
            prefix_name = prefix_name + "time_" if is_time_weights else prefix_name
            prefix_name = prefix_name + "velocities_"
            prefix_name = (
                prefix_name + "occlusions_" if is_occlusions_weights else prefix_name
            )

            if prediction_method == "NN":
                prefix_name = prefix_name + "SimpleDNN"
            elif prediction_method == "Transformer":
                prefix_name = prefix_name + "ScratchTransformer"

            model_path = f"{pkg_path}/scripts/mml_network/models/{prefix_name}.pth"
        elif prediction_method == "DMMN":
            model_path = f"/home/basestation/base_ws/src/mml_guidance/mml_guidance/mml_network/models/online_model.pth"

        # Initialize with ROS2 time
        initial_time = clock.now().seconds_nanoseconds()[0]

        # Call parent constructor
        super().__init__(
            num_particles=num_particles,
            prediction_method=prediction_method,
            is_sim=is_sim,
            drone_height=drone_height,
            model_path=model_path,
            initial_time=initial_time,
        )

    def get_current_time(self):
        """Override to use ROS2 clock"""
        return clock.now().seconds_nanoseconds()[0] - self.initial_time

    def save_model(self, save_path=None):
        """Override to use ROS2 package paths"""
        if save_path is None:
            pkg_path = get_package_share_directory("mml_guidance")
            save_path = f"{pkg_path}/scripts/mml_network/models/online_model.pth"

        if self.prediction_method in {"NN", "Transformer"}:
            super().save_model(save_path)

            # Also save training data to ROS2 package location
            csv_file = f"{pkg_path}/sim_data/training_data/online_data.csv"
            if not os.path.exists(csv_file):
                os.makedirs(os.path.dirname(csv_file), exist_ok=True)

            filled_elements = self.state_history.shape[0]  # Use full buffer
            X_train, y_train = self.convert_state_history_to_training_batches(
                filled_elements, all_data=True
            )

            if X_train is not None:
                import numpy as np

                X_train_flattened = X_train.reshape(X_train.shape[0], -1)
                state_history_flattened = np.concatenate(
                    (
                        X_train_flattened,
                        y_train,
                        np.zeros((X_train_flattened.shape[0], 1)),
                    ),
                    axis=1,
                )
                np.savetxt(csv_file, state_history_flattened, delimiter=",")
                print(f"Data saved to {csv_file}")
        elif self.prediction_method == "DMMN":
            self.motion_model.saveModel(
                time=clock.now().seconds_nanoseconds()[0], savePath=save_path
            )

        print(f"Model saved to {save_path}")
