"""Prediction and model configuration for the production particle filter.

The filter owns mutable state; this collaborator uses that live state so tuning,
legacy attribute access, and deep-copied tracks retain their original semantics.
The controlled PF, tutorial PF, and DiffPF algorithms remain separate.
"""
import numpy as np
import torch
from mml_core.motion.motion_model_network import MotionModel
from mml_core.motion.simple_dnn import SimpleDNN
from mml_core.motion.scratch_transformer import ScratchTransformer


class ParticlePrediction:
    def __init__(self, particle_filter):
        self.filter = particle_filter

    def configure_model(self, model_path):
        pf = self.filter
        if pf.prediction_method in {
            "NN", "Transformer", "CFM", "CFM-AS", "NF", "Diffusion"
        }:
            pf.N_th = 5   # overridden for CFM after checkpoint is loaded below
            pf.is_velocity = True
            is_occlusions_weights = True
            pf.is_time_weights = False
            input_dim = 3 if pf.is_time_weights else 2 
            pf.nn_input_size = pf.N_th * input_dim
            pf.device = "cuda" if torch.cuda.is_available() else "cpu"
            prefix_name = "noisy_5_v2_"  # "connected_graph_noisy_5_" or "noisy_5_" for pretrained on road network
            prefix_name = prefix_name + "time_" if pf.is_time_weights else prefix_name
            prefix_name = (
                prefix_name + "velocities_"
                if pf.is_velocity
                else prefix_name + "position_"
            )
            prefix_name = (
                prefix_name + "occlusions_" if is_occlusions_weights else prefix_name
            )

            if pf.prediction_method == "NN":
                prefix_name = prefix_name + "SimpleDNN"
                pf.motion_model = SimpleDNN(
                    input_size=pf.nn_input_size,
                    nodes_per_layer=80,
                    num_layers=2,
                    output_size=2,
                    activation_fn="relu",
                )
            elif pf.prediction_method == "Transformer":
                prefix_name = prefix_name + "ScratchTransformer"
                pf.motion_model = ScratchTransformer(
                    input_dim=input_dim,
                    block_size=pf.N_th,
                    n_embed=20,
                    n_head=8,
                    n_layer=2,
                    hidden_dim=20,
                )
            elif pf.prediction_method in {"CFM", "CFM-AS"}:
                
                # ── Conditional Flow Matching motion model ────────────────────
                # load_cfm_model() reads ALL hyperparameters from the checkpoint
                # automatically — no need to pass n_embed/n_head etc. manually.
                from mml_core.motion.cfm_motion_model import load_cfm_model
                if model_path is None:
                    raise ValueError(
                        f"{pf.requested_prediction_method} requires a CFM checkpoint."
                    )
                pf.motion_model = load_cfm_model(
                    model_path = model_path,
                    input_dim  = 2,
                    block_size = pf.N_th,  # fallback — overridden by checkpoint
                    # Fallback defaults — overridden by checkpoint config automatically:
                    n_embed    = 64,
                    n_head     = 8,
                    n_layer    = 2,
                    vel_hidden = 128,
                    N_FE       = 2,
                )
                # Sync N_th from the checkpoint — the loaded model is authoritative.
                pf.N_th = pf.motion_model._cfm.block_size
                pf.nn_input_size = pf.N_th * input_dim
                if pf.requested_prediction_method in {"PF+CFM", "PF+CFM-AS"}:
                    pf.motion_model._cfm.anchor_sigma = 1.0
                pf.cfm_base_sigma = float(pf.motion_model._cfm.anchor_sigma)
                print(f"[ParticleFilter] CFM loaded from {model_path}  "
                      f"(N_th={pf.N_th}, sigma0={pf.cfm_base_sigma})")
                pf.is_velocity = True
            elif pf.prediction_method == "NF":
                from mml_core.motion.nf_motion_model import load_nf_model
                if model_path is None:
                    raise ValueError("PF+NF requires an NF checkpoint.")
                pf.motion_model = load_nf_model(
                    model_path=model_path,
                    input_dim=2,
                    block_size=pf.N_th,
                )
                pf.N_th = pf.motion_model._nf.block_size
                pf.nn_input_size = pf.N_th * input_dim
                print(f"[ParticleFilter] NF loaded from {model_path}  "
                      f"(N_th={pf.N_th})")
                pf.is_velocity = True
            elif pf.prediction_method == "Diffusion":
                from mml_core.motion.diffusion_motion_model import load_diffusion_model
                if model_path is None:
                    raise ValueError(
                        "PF+Diffusion requires a diffusion motion-model checkpoint."
                    )
                pf.motion_model = load_diffusion_model(
                    model_path=model_path,
                    input_dim=2,
                    block_size=pf.N_th,
                )
                pf.N_th = pf.motion_model._diffusion.block_size
                pf.nn_input_size = pf.N_th * input_dim
                print(
                    f"[ParticleFilter] Diffusion loaded from {model_path}  "
                    f"(N_th={pf.N_th}, T={pf.motion_model._diffusion.T})"
                )
                pf.is_velocity = True

            # Load model if path is provided
            if model_path and pf.prediction_method not in {
                "CFM", "CFM-AS", "NF", "Diffusion"
            }:
                # CFM is already fully loaded inside load_cfm_model() above.
                # Skip this block for CFM — torch.load() returns the full checkpoint
                # dict {"state_dict":..., "config":...} which is NOT a raw state_dict.
                try:
                    pf.motion_model.load_state_dict(
                        torch.load(model_path, map_location=pf.device)
                    )
                except FileNotFoundError:
                    print(f"Error: The file '{model_path}' does not exist. Change to the correct directory.")
            elif pf.prediction_method not in {
                "CFM", "CFM-AS", "NF", "Diffusion"
            }:
                print("No model path provided for NN/Transformer. Using untrained model.")
        else:
            pf.N_th = 1 if pf.prediction_method == "DiffPF" else 2

    def configure_dynamics(self, model_path):
        pf = self.filter
        deg2rad = lambda deg: np.pi * deg / 180
        if pf.prediction_method == "Velocity":
            pf.Nx = 4  # number of states [x, y, vx, vy]
            pf.vmax = 0.4  # m/s maximum velocity
            pf.best_measurement_covariance = np.diag([0.006, 0.006])
            # Process noise parameters - improved from reference code
            pf.proc_pos_std = 0.05  # m per sqrt(s) - position process noise
            pf.proc_vel_std = 0.15   # m/s per sqrt(s) - velocity process noise
            pf.process_covariance = np.diag([pf.proc_pos_std**2, pf.proc_pos_std**2, 
                                             pf.proc_vel_std**2, pf.proc_vel_std**2])
        elif pf.prediction_method == "Unicycle":
            pf.Nx = 3
            pf.best_measurement_covariance = np.diag([0.01, 0.01, deg2rad(5)])
            pf.process_covariance = np.diag([0.8, 0.8, deg2rad(5.)])
        elif pf.prediction_method in {"KF", "DMMN"}:
            pf.Nx = 2
            pf.best_measurement_covariance = np.diag([0.006, 0.006])
            pf.process_covariance = np.diag([0.0001, 0.0001, 0.00001, 0.00001])
        elif pf.prediction_method == "DiffPF":
            pf.Nx = 2
            pf.best_measurement_covariance = np.diag([0.006, 0.006])
            pf.process_covariance = np.diag([0.001, 0.001])
            if model_path is None:
                raise ValueError("DiffPF requires a trained DiffPF checkpoint.")
            from mml_core.tracking.diffpf_tracker import load_diffpf_tracker
            pf.diffpf_tracker = load_diffpf_tracker(
                model_path,
                n_particles=pf.N,
            )
            # The network does not depend on the number of posterior samples,
            # so use the common full-system particle count after loading.
            pf.diffpf_tracker.n_particles = pf.N
            print(f"[ParticleFilter] DiffPF loaded from {model_path}  "
                  f"(n_particles={pf.N})")

        if pf.prediction_method == "DMMN":
            inputSize = 2  # number of inputs to NN
            gamma = 0.1  # a learning for online learning
            k1 = 4.0  # a learning for online learning
            # NN parameters
            numberHiddenLayers = 2  # number of hidden layers
            hiddenSize = 6  # size of each hidden layer
            outputSize = 5  # size of the output of the NN
            probHiddenDrop = 0.0  # dropout probability for hidden layers
            useAttention = False  # use attention at output layer

            # learning parameters
            alpha = 0.001  # learning rate for SGD
            memorySize = 150  # number of data points to save for each training
            batchSize = 20  # size of a batch for each training epoch, should be multiple of memory size
            numberEpochs = 25  # 25  # number of iterations for each training
            minDistance = 0.005  # minimum distance between any two data points

            pf.motion_model = MotionModel(
                alpha,
                numberHiddenLayers,
                inputSize,
                hiddenSize,
                outputSize,
                probHiddenDrop,
                useAttention,
                memorySize,
                batchSize,
                numberEpochs,
                minDistance,
                gamma,
                k1,
            )

            # Load model if path is provided
            if model_path:
                pf.motion_model.loadModel(model_path)

        if pf.prediction_method in {
            "NN", "Transformer", "CFM", "CFM-AS", "NF", "Diffusion"
        }:
            pf.Nx = 2
            pf.best_measurement_covariance = np.diag([0.006, 0.006])
            pf.process_covariance = np.diag([0.001, 0.001])
            buffer_size = (
                300  # large but not infinite to not have dynamic memory allocation
            )
            pf.max_batch_size = 64
            assert (
                pf.N_th + pf.max_batch_size < buffer_size
            ), "Buffer size too small"
        else:
            buffer_size = 1

        return buffer_size

    def predict_mml(self, particles, dt_history):
        """Propagates the current particles through the motion model learned with the
        neural network.
        Input: N_th (number of time histories) sets of particles up to time k-1
        Output: N_th sets of propagated particles up to time k
        """
        pf = self.filter
        if pf.is_time_weights:
            tiled_dt_hist = np.tile(
                dt_history.reshape(-1, 1), (1, particles.shape[1])
            ).reshape(-1, particles.shape[1], 1)
            particles = np.concatenate((particles, tiled_dt_hist), axis=2)

        next_parts = pf.motion_model.predict(
            np.transpose(particles[:, :, :], (1, 0, 2)).reshape(
                particles.shape[1], pf.nn_input_size
            )
        )

        if pf.is_velocity:
            next_parts = (
                particles[-1, :, :2] + next_parts * dt_history[-1]
            )  # x(t+1) = x(t) + v(t) * dt

        # roll particles in time, add latest prediction
        particles = np.concatenate(
            (
                particles[1:, :, :2],
                next_parts.reshape(1, next_parts.shape[0], next_parts.shape[1]),
            ),
            axis=0,
        )
        # important: will not work without this
        particles[-1, :, :] = pf.add_noise(
            particles[-1, :, :], 2 * pf.process_covariance
        )
        return particles

    def predict_cfm(self, particles, dt_history):
        """
        Propagate particles with a stochastic learned velocity proposal.

        CFM and NF both expose propose(context) and a training timestep.
        Using the checkpoint timestep keeps position histories in the
        distribution used to train the velocity model.
        """
        pf = self.filter
        # (N_th, N, 2) -> (N, N_th * 2)
        c_flat = np.transpose(particles[:, :, :2], (1, 0, 2)).reshape(pf.N, pf.N_th * 2)

        # Each particle gets an independent stochastic velocity draw.
        # Clipping is applied inside CFMMotionModel.propose() itself.
        v_pred = pf.motion_model.propose(c_flat)  # (N, 2)  m/s at training dt scale

        # Use the dt the model was trained at to keep position update in distribution.
        train_dt = float(getattr(pf.motion_model, "dt", 0.033))

        particles[:-1, :, :] = particles[1:, :, :]
        particles[-1, :, :2] = particles[-2, :, :2] + v_pred * train_dt

        particles[-1, :, :2] = pf.add_noise(
            particles[-1, :, :2],
            pf.process_covariance[:2, :2],
        )
        return particles

    def predict(
        self,
        particles,
        dt,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        """Uses the unicycle model plus noise or the to propagate the belief in the system state.
        The position and orientation of the particles are updated according to the linear
        and angular velocity of the turtlebot.
        In the Velocity method, the particles are propagated according to the velocity of each particle.
        Input: State of the particles at time k-1
        Output: Predicted (propagated) state of the particles up to time k
        """
        pf = self.filter
        particles[:-1, :, :] = particles[1:, :, :]  # shift particles in time
        if pf.prediction_method == "Unicycle":
            delta_theta = angular_velocity[0] * dt
            particles[-1, :, 2] = (
                particles[-2, :, 2]
                + delta_theta
                + (delta_theta / (2 * np.pi))
                * pf.add_noise(
                    np.zeros(particles.shape[1]),
                    pf.process_covariance[2, 2],
                    size=particles.shape[1],
                )
            )

            for ii in range(particles.shape[1]):
                if np.abs(particles[-1, ii, 2]) > np.pi:
                    # Wraps angle
                    particles[-1, ii, 2] = (
                        particles[-1, ii, 2] - np.sign(particles[-1, ii, 2]) * 2 * np.pi
                    )

            norm_lin_vel = np.linalg.norm(linear_velocity)
            delta_distance = norm_lin_vel * dt + norm_lin_vel * dt * pf.add_noise(
                0, pf.process_covariance[0, 0], size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2]
                + np.array(
                    [
                        delta_distance * np.cos(particles[-1, :, 2]),
                        delta_distance * np.sin(particles[-1, :, 2]),
                    ]
                ).T
            )
        elif pf.prediction_method == "Velocity":
            dt = max(dt, 1e-6)  # Avoid division by zero
            
            # Position update: x = x + vx*dt + noise
            pos_noise = pf.add_noise(
                np.array([0, 0]), 
                np.diag([pf.proc_pos_std**2, pf.proc_pos_std**2]) * dt,
                size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2] + 
                particles[-1, :, 2:4] * dt + 
                pos_noise
            )
            
            # Velocity update: vx = vx + noise
            vel_noise = pf.add_noise(
                np.array([0, 0]), 
                np.diag([pf.proc_vel_std**2, pf.proc_vel_std**2]) * dt,
                size=particles.shape[1]
            )
            particles[-1, :, 2:4] = particles[-1, :, 2:4] + vel_noise
            
            # Velocity bounds to prevent unrealistic velocities
            particles[-1, :, 2:4] = np.clip(particles[-1, :, 2:4], -pf.vmax, pf.vmax)

        return particles
