from datetime import datetime, timedelta
import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator


class KalmanFilterGNNAssociator:
    def __init__(self, process_noise=1.0, measurement_noise=0.8, max_measurement_noise=4.0):
        """
        Initialize the multi-target tracker with Kalman filters and data association.
        
        Args:
            process_noise: Process noise for the transition model
            measurement_noise: Base measurement noise
            max_measurement_noise: Maximum measurement noise for low confidence detections
        """
        # self.process_covariance = np.diag([process_noise] * 2)
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(process_noise), ConstantVelocity(process_noise)]
        )
        self.measurement_noise = measurement_noise
        self.max_measurement_noise = max_measurement_noise
        self.measurement_covariance = np.diag([measurement_noise] * 2)
        self.measurement_model = LinearGaussian(
            ndim_state=4, mapping=(0, 2), noise_covar=self.measurement_covariance
        )
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(self.measurement_model)
        self.hypothesizer = DistanceHypothesiser(
            self.predictor, self.updater, measure=Mahalanobis(), missed_distance=3
        )
        self.data_associator = GNNWith2DAssignment(self.hypothesizer)

        self.deleter = CovarianceBasedDeleter(covar_trace_thresh=9)
        self.initiator = MultiMeasurementInitiator(
            prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5])),
            measurement_model=self.measurement_model,
            updater=self.updater,
            data_associator=self.data_associator,
            deleter=self.deleter,
            min_points=1,
        )

        self.tracks = set()
        self.start_time = datetime.now().replace(microsecond=0)
        self.is_update = False
        self.current_time = self.start_time

    def update_time(self, time_step=None):
        """Update the current time for tracking."""
        if time_step is not None:
            if isinstance(time_step, (int, float)):
                # Convert float time step (seconds since start) to datetime
                self.current_time = self.start_time + timedelta(seconds=float(time_step))
            else:
                # Assume it's already a datetime object
                self.current_time = time_step
        else:
            self.current_time = datetime.now().replace(microsecond=0)

    def predict_tracks(self, tracks=None, steps_ahead=1, **kwargs):
        """
        Perform prediction step for all tracks without measurements.
        This can be called when no new detections are available.
        """
        if tracks is None:
            is_future_estimates = False
            tracks = self.tracks
            prediction_time = self.current_time
        else:
            is_future_estimates = True
            # For future estimates, predict one time step ahead
            if isinstance(self.current_time, datetime):
                prediction_time = self.current_time + timedelta(seconds=0.1 * steps_ahead)  # 0.1 second step
            else:
                prediction_time = self.current_time + 0.1 * steps_ahead  # 0.1 time unit step

        for track in tracks:
            if len(track) > 0:
                # Predict track state to prediction time
                prediction = self.predictor.predict(track, timestamp=prediction_time)
                track.append(prediction)
                
        # Delete tracks with high covariance after prediction
        # print("track ids before deletion: ", [track.id[-1] for track in self.tracks])
        # print("state covar before deletion: ", [round(np.trace(track[-1].covar), 1) for track in self.tracks])
        N_before = len(self.tracks)
        tracks -= self.deleter.delete_tracks(tracks)
        if not is_future_estimates:
            print(f"Deleted {N_before - len(self.tracks)} tracks due to high covariance.") if N_before != len(self.tracks) else None
            self.tracks = tracks
        else:
            return tracks

    def process_detections(self, detections, confidences=None, future_tracks=None, **kwargs):
        """
        Process new detections and update tracks.
        
        Args:
            detections: List of [x, y] positions or numpy array of shape (N, 2)
            confidences: List of confidence values (0-1) for each detection, optional
        """
        if len(detections) == 0:
            # No detections - just predict existing tracks
            if not future_tracks:
                self.predict_tracks()
            else:
                future_tracks = self.predict_tracks(tracks=future_tracks, **kwargs)
            return
            
        # Convert detections to numpy array if needed
        if isinstance(detections, list):
            detections = np.array(detections)
            
        # Create Detection objects
        measurements = [
            Detection(
                np.array([[det[0]], [det[1]]]), timestamp=self.current_time
            )
            for det in detections
        ]
        
        # Use default confidence if not provided
        if confidences is None:
            confidences = [1.0] * len(detections)
            
        if not future_tracks:
            self.process_measurements(measurements, confidences)
        else:
            return self.process_measurements(measurements, confidences, future_tracks=future_tracks)

    def inv_proportional_noise(self, confidence):
        """ Returns a measurement covariance matrix that is inversely proportional to the confidence level.
            The mapping is linear with min_confidence confidence being the maximum measurement noise and 1 confidence being the
            ideal measurement noise measurement_noise.
        Args:
            confidence: The confidence level between min_confidence and 1."""
        min_confidence = 0.1  # from learned detection model
        m = ((self.measurement_noise - self.max_measurement_noise)/(1 - min_confidence))
        return np.diag([m*confidence + self.measurement_noise - m]*2)

    def process_measurements(self, measurements, confidences, future_tracks=None):
        """Process measurements and update tracks with data association."""
        # Associate measurements with tracks
        tracks = self.tracks if not future_tracks else future_tracks.copy()
        is_future_estimates = True if future_tracks else False
        hypotheses = self.data_associator.associate(
            tracks, measurements, self.current_time
        )
        associated_measurements = set()

        # Update measurement model noise based on confidence
        if confidences and len(confidences) > 0:
            self.updater.measurement_model.noise_covar = self.inv_proportional_noise(confidences[0])

        # print("Number of tracks: ", len(tracks))
        track_counter = 0
        for track in tracks:
            track_hypotheses = hypotheses[track]
            if track_hypotheses.measurement:
                post = self.updater.update(track_hypotheses)
                track.append(post)
                associated_measurements.add(track_hypotheses.measurement)
            else:
                track.append(track_hypotheses.prediction)
            
            track_counter += 1
            # print("track: ", np.array([track.state_vector[0], track.state_vector[2]]).T)
            
        # Delete tracks with high covariance
        tracks -= self.deleter.delete_tracks(tracks)

        # Initiate new tracks from measurements
        new_measurements = set(measurements) - associated_measurements
        tracks |= self.initiator.initiate(new_measurements, self.current_time)
        if not future_tracks:
            self.tracks = tracks
        else:
            return tracks


    def get_track_states(self, tracks=None):
        """
        Get current track states as a list of [x, y, vx, vy] arrays.
        
        Returns:
            List of numpy arrays, each containing [x, y, vx, vy] for a track
        """
        if tracks is None:
            tracks = self.tracks
        track_states = []
        for track in tracks:
            if len(track) > 0:
                latest_state = track[-1]
                state_vector = latest_state.state_vector.flatten()
                # switch from [x, vx, y, vy] to [x, y, vx, vy]
                state_vector = np.array([state_vector[0], state_vector[2], state_vector[1], state_vector[3]])
                track_states.append(state_vector)
        return track_states

    def get_track_covariances(self, tracks=None):
        """
        Get current track covariances as a list of covariance matrices.
        The covariance takes the form:
        [[var_x, cov_xy, var_vx, cov_xvx],
         [cov_yx, var_y, cov_yvx, cov_yvy],
         [var_vx, cov_vxy, var_vx, cov_vxvy],
         [cov_vyx, var_vy, cov_vyx, var_vy]]
        
        Returns:
            List of numpy arrays, each containing the covariance matrix for a track
        """
        if tracks is None:
            tracks = self.tracks
        track_covariances = []
        for track in tracks:
            if len(track) > 0:
                latest_state = track[-1]
                covar = latest_state.covar
                # switch from [x, vx, y, vy] to [x, y, vx, vy]
                covar = np.array([[covar[0,0], covar[0,2], covar[0,1], covar[0,3]],
                                  [covar[2,0], covar[2,2], covar[2,1], covar[2,3]],
                                  [covar[1,0], covar[1,2], covar[1,1], covar[1,3]],
                                  [covar[3,0], covar[3,2], covar[3,1], covar[3,3]]])
                track_covariances.append(covar * 0.25)  # scale down 
        return np.array(track_covariances)
