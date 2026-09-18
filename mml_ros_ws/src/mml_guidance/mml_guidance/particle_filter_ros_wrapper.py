#!/usr/bin/env python3
import os
from pathlib import Path
from mml_core.config import project_root, resolve_model_path
from rclpy.clock import Clock, ClockType
from ament_index_python.packages import get_package_share_directory
from mml_core.tracking.particle_filter import ParticleFilter

# ROS2 Clock for timing
clock = Clock(clock_type=ClockType.ROS_TIME)


class ParticleFilterROS2(ParticleFilter):
    def __init__(
        self, num_particles=10, prediction_method="NN", is_sim=False,
        drone_height=2.0, model_path=None,
    ):
        # Keep ROS-only arguments in this adapter, not in the numerical filter.
        self.is_sim = is_sim
        if model_path:
            model_path = os.path.abspath(os.path.expanduser(model_path))
        else:
            model_path = resolve_model_path(str(project_root()), prediction_method)
            if model_path and not os.path.isfile(model_path):
                package_path = Path(get_package_share_directory("mml_guidance"))
                subdir = "mml_guidance/mml_network/models" if prediction_method in {"Transformer", "MultiPFTra"} else "models"
                model_path = str(package_path / subdir / Path(model_path).name)
            if prediction_method in {"NN", "DMMN"}:
                raise ValueError(f"{prediction_method} requires an explicit compatible model_path.")
        initial_time = clock.now().seconds_nanoseconds()[0]
        self.initial_time = initial_time
        super().__init__(
            num_particles=num_particles,
            prediction_method=prediction_method,
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
            save_path = str(Path.cwd() / "models/online_model.pth")
        save_path = os.path.abspath(os.path.expanduser(save_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.prediction_method in {"NN", "Transformer"}:
            super().save_model(save_path)

            # Also save training data to ROS2 package location
            csv_file = str(Path(save_path).parent / "online_data.csv")
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
