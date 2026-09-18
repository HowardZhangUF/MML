"""Information scoring and goal selection using the guidance object's live state."""
import copy
import numpy as np
import scipy.stats as stats


class InformationGuidance:
    def __init__(self, guidance):
        self.guidance = guidance

    def current_entropy(self, event=None) -> None:
        """Compute the entropy of the current distribution of particles
        Equation changes if there is a measurement available. See paper for reference
        We sample the particle distribution to reduce computation time.
        Output: Hp_t (float) - entropy of the current distribution"""
        guidance = self.guidance

        prev_Hp = np.copy(guidance.Hp_t)


        # Normalize weights
        guidance.sampled_weights = (
            guidance.sampled_weights / np.sum(guidance.sampled_weights)
            if np.sum(guidance.sampled_weights) > 0
            else guidance.sampled_weights
        )
        sampled_prev_weights = np.copy(guidance.filter.prev_weights[guidance.sampled_index])

        if guidance.filter.is_update:
            Hp_t = guidance.entropy_particle(
                guidance.sampled_particles[-2],
                np.copy(sampled_prev_weights),
                guidance.sampled_particles[-1],
                np.copy(guidance.sampled_weights),
                guidance.noisy_turtle_pose,
            )
        else:
            Hp_t = guidance.entropy_particle(
                guidance.sampled_particles[-2],
                np.copy(
                    guidance.sampled_weights
                ),  # current weights are the (t-1) weights because no update
                guidance.sampled_particles[-1],
            )

        Hp_t = (
            prev_Hp if not np.isfinite(Hp_t) else Hp_t
        )  # if really bad value, keep the previous one
        if not np.array_equal(guidance.prev_Hp, np.ones((5, 1))):
            guidance.Hp_t = guidance.reject_spikes(
                guidance.prev_Hp,
                Hp_t,
                threshold_factor=3,
            )

        # update entropy history
        guidance.prev_Hp = np.roll(guidance.prev_Hp, -1, axis=0)
        guidance.prev_Hp[-1, :2] = Hp_t

    def propagate_particles(self, future_parts, dt=0.33, particle_filter=None):
        guidance = self.guidance
        particle_filter = particle_filter or guidance.filter
        prediction_method = particle_filter.prediction_method
        if prediction_method in {"CFM", "CFM-AS", "NF", "Diffusion"}:
            future_parts = particle_filter.predict_cfm(
                np.copy(future_parts),
                np.ones(future_parts.shape[0]) * dt,
            )
        elif prediction_method in {"NN", "Transformer"}:
            future_parts = particle_filter.predict_mml(
                np.copy(future_parts), np.ones(future_parts.shape[0]) * dt
            )
        elif prediction_method == "Unicycle":
            future_parts = particle_filter.predict(
                future_parts,
                dt,
                angular_velocity=guidance.angular_velocity,
                linear_velocity=guidance.linear_velocity,
            )
        elif prediction_method == "Velocity":
            future_parts = particle_filter.predict(future_parts, dt)
        return future_parts

    def information_driven_guidance(self, future_parts):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the  position of the particle
        that maximizes the EER.
        Output:
        eer_particle: the index of the particle that maximizes the EER which
        we propagate to choose the goal position
        """
        guidance = self.guidance
        # Initialize variables
        future_weight = np.zeros((guidance.N_s, guidance.N_s))
        Hp_k = np.zeros(guidance.N_s)  # partial entropy
        EER = np.zeros(guidance.N_s)  # Expected Entropy Reduction

        # Future possible measurements
        # TODO: implement N_m sampled measurements (assumption: N_m = 1)
        z_hat = guidance.filter.add_noise(
            future_parts[-1, :, : guidance.filter.measurement_covariance.shape[0]],
            guidance.filter.measurement_covariance,
        )

        # p(z_{k+K} | x_{k+K})
        likelihood = guidance.filter.likelihood(future_parts[-1], z_hat)
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(guidance.N_s):
            k_fov = guidance.construct_FOV(z_hat[jj])  # assuming we are on top of measurement
            # checking for measurement outside of fov or in occlusion
            if guidance.is_in_FOV(z_hat[jj], k_fov) and not guidance.occlusions.in_occlusion(
                z_hat[jj]
            ):
                future_weight[:, jj] = guidance.filter.update(
                    guidance.sampled_weights, future_parts, z_hat[jj]
                )

                # H (x_{k+K} | \hat{z}_{k+K})
                Hp_k[jj] = guidance.entropy_particle(
                    future_parts[-2],
                    np.copy(guidance.sampled_weights),
                    future_parts[-1],
                    future_weight[:, jj],
                    z_hat[jj],
                )
            else:
                Hp_k[jj] = guidance.entropy_particle(
                    future_parts[-2],
                    np.copy(guidance.sampled_weights),
                    future_parts[-1],
                )

            # Information Gain
            EER[jj] = guidance.Hp_t - (Hp_k[jj] )* likelihood[jj]

        guidance.eer_particle = np.argmax(EER)

    def info_driven_guidance_multi_kf(self, future_estimates):
        """Compute the current entropy and future entropy using future gaussian estimates"""
        guidance = self.guidance
        assert len(future_estimates) > 0, "No future estimates provided for info-driven guidance"
        if len(future_estimates) == 1:
            return guidance.multi_filter.get_track_states(tracks=future_estimates)[0][:2]
        Hp_k = [0.0] * len(future_estimates)  # partial entropy
        for ii, estimate in enumerate(future_estimates):
            # possible future measurement
            estimate_pos = guidance.multi_filter.get_track_states(tracks=[estimate])[0][:2]
            z_hat = guidance.filter.add_noise(
                estimate_pos, guidance.filter.measurement_covariance[:2, :2],
            )

            if guidance.is_in_FOV(z_hat, guidance.construct_FOV(estimate_pos)) and not guidance.occlusions.in_occlusion(z_hat):
                # posterior tracks after update with possible measurement
                future_posterior_tracks = guidance.multi_filter.process_detections([z_hat], confidences=None, future_tracks=future_estimates)
            else:
                future_posterior_tracks = future_estimates
            if len(future_posterior_tracks) > 0:
                track_states = guidance.multi_filter.get_track_states(tracks=future_posterior_tracks)
                track_covs = guidance.multi_filter.get_track_covariances(tracks=future_posterior_tracks)
                for jj in range(len(track_states)):
                    Hp_k[ii] += guidance.entropy_gaussian(track_states[jj], track_covs[jj])
            else:
                Hp_k[ii] = 1e6  # very high entropy if no tracks
        
        # choose the estimate that minimizes the entropy
        eer_track = np.argmin(Hp_k)  # I do not think I need to multiply by likelihood because it is constant wrt the action
        print("entropies:", [f"{float(e):.3f}" for e in Hp_k])
        print(f"goal track: {eer_track}")
        for ii, estimate in enumerate(future_estimates):
            if ii == eer_track:
                return guidance.multi_filter.get_track_states(tracks=[estimate])[0][:2]

    def info_driven_guidance_multi_pf(self, future_tracks):
        """Compute the current entropy and future entropy using future particle filter tracks"""
        guidance = self.guidance
        assert len(future_tracks) > 0, "No future tracks provided for info-driven guidance"
        if len(future_tracks) == 1:
            # Only one track, follow it
            return future_tracks[0].weighted_mean[:2]
        
        Hp_k = [0.0] * len(future_tracks)  # partial entropy
        for ii, track in enumerate(future_tracks):
            # possible future measurement
            estimate_pos = track.weighted_mean[:2]
            z_hat = guidance.filter.add_noise(
                estimate_pos, guidance.filter.measurement_covariance[:2, :2],
            )

            if guidance.is_in_FOV(z_hat, guidance.construct_FOV(estimate_pos)) and not guidance.occlusions.in_occlusion(z_hat):
                # Simulate update with possible measurement
                # Create a copy of the track for future prediction
                future_track_copy = copy.deepcopy(track)
                future_track_copy.pf.weights = future_track_copy.pf.update( \
                    future_track_copy.pf.weights, future_track_copy.pf.particles, z_hat)
                
                # Compute entropy based on particle spread after update
                particles = future_track_copy.pf.particles[-1, :, :2]
                weights = future_track_copy.pf.weights
                
                # Weighted covariance
                mean = np.average(particles, weights=weights, axis=0)
                diff = particles - mean
                cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                               weights=weights, axis=0)
                
                # Entropy = 0.5 * log((2*pi*e)^n * |Sigma|)
                det_cov = np.linalg.det(cov + 1e-6 * np.eye(2))
                Hp_k[ii] = 0.5 * np.log((2 * np.pi * np.e) ** 2 * max(det_cov, 1e-6))
            else:
                # No measurement update, use current entropy
                particles = track.pf.particles[-1, :, :2]
                weights = track.pf.weights
                
                mean = np.average(particles, weights=weights, axis=0)
                diff = particles - mean
                cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                               weights=weights, axis=0)
                
                det_cov = np.linalg.det(cov + 1e-6 * np.eye(2))
                Hp_k[ii] = 0.5 * np.log((2 * np.pi * np.e) ** 2 * max(det_cov, 1e-6))
        
        # Choose the track that minimizes the entropy
        eer_track = np.argmin(Hp_k)
        # print("particle filter entropies:", [f"{float(e):.3f}" for e in Hp_k])
        # print(f"selected track: {eer_track}")

        return future_tracks[eer_track].weighted_mean[:2]

    def entropy_gaussian(self, mean, cov):
        """Compute the entropy of a gaussian distribution
        H = 0.5 * ln((2*pi*e)^n * |Sigma|)
        where n is the dimension of the gaussian and Sigma is the covariance matrix
        Output: entropy (float)"""
        guidance = self.guidance
        n = mean.shape[0]
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            det_cov = 1e-6  # numerical stability
        entropy = 0.5 * np.log((2 * np.pi * np.e) ** n * det_cov)
        return entropy

    def entropy_particle(
        self,
        prev_particles,
        prev_wgts,
        particles,
        wgts=np.array([]),
        z_meas=np.array([]),
    ) -> np.int64:
        """Compute the entropy of the particle distribution based on the equation in the
        paper: Y. Boers, H. Driessen, A. Bagchi, and P. Mandal, 'Particle filter based entropy'
        There are two computations, one for the case where the measurement is inside the fov
        and there is an update step before, and one where the measurement is outside the fov.
        Output:
            entropy (numpy.int64)
        """
        guidance = self.guidance
        if wgts.size > 0:
            # likelihod of measurement p(zt|xt)
            # (how likely is each of the particles in the gaussian of the measurement)
            like_meas = stats.multivariate_normal.pdf(
                x=particles[:, :2],
                mean=z_meas[:2],
                cov=guidance.filter.measurement_covariance[:2, :2],
            )

            # likelihood of particle p(xt|xt-1)
            part_len, _ = particles.shape
            process_part_like = np.zeros(part_len)

            for ii in range(part_len):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=guidance.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts[ii])

            # Numerical stability
            cutoff = 1e-4
            like_meas[like_meas < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan
            # remove the nans from the likelihoods
            # like_meas = like_meas[~np.isnan(like_meas)]
            process_part_like[process_part_like < cutoff] = np.nan

            product = like_meas * prev_wgts
            notnans = product[~np.isnan(product)]
            notnans[notnans < cutoff * 0.01] = np.nan
            product[~np.isnan(product)] = notnans
            first_term = np.log(np.nansum(product))
            # first_term = first_term if np.isfinite(first_term) else 0.0
            # second_term = np.nansum(np.log(prev_wgts)*weights)
            # third_term = np.nansum(weights*np.log(like_meas))
            # fourth_term = np.nansum(weights*np.log(process_part_like))

            entropy = (
                first_term
                - np.nansum(np.log(prev_wgts) * wgts)
                - np.nansum(wgts * np.log(like_meas))
                - np.nansum(wgts * np.log(process_part_like))
            )

            # if np.abs(entropy) > 30:  # debugging
            # print("\nEntropy term went bad :(")
            # print("second term: ", np.nansum(np.log(prev_wgts) * wgts))
            # print("third term: ", np.nansum(wgts * np.log(like_meas)))
            # print("fourth term: ", np.nansum(wgts * np.log(process_part_like)))

            if np.isinf(first_term):
                print("first term of entropy is -inf. Likelihood is very small")
        else:
            # likelihood of particle p(xt|xt-1)
            part_len2, _ = prev_particles.shape
            process_part_like = np.zeros(part_len2)
            for ii in range(part_len2):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=guidance.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts)

            # Numerical stability
            cutoff = 1e-4
            process_part_like[process_part_like < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan

            entropy = -np.nansum(prev_wgts * np.log(process_part_like))

        return entropy
