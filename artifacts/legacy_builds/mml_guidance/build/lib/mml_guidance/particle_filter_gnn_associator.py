import numpy as np
from .ParticleFilter import ParticleFilter


class ParticleFilterTrack:
    """
    A single target track using particle filter for state estimation.
    Each track maintains its own particle filter instance.
    """
    def __init__(self, track_id, initial_detection, initial_time, 
                 num_particles=200, prediction_method="Unicycle", model_path=None, 
                 drone_height=2.0):
        self.id = track_id
        self.last_update_time = initial_time
        self.hits = 1  # Number of successful updates
        self.missed_detections = 0
        self.is_bad_weights = False 
        
        # Initialize particle filter for this track
        self.pf = ParticleFilter(
            num_particles=num_particles,
            prediction_method=prediction_method,
            initial_time=initial_time,
            initial_detection=initial_detection,
            model_path=model_path,
            drone_height=drone_height
        )
        
        self._update_state_estimates()

    def _update_state_estimates(self):
        """Update the state estimate from particle filter"""
        # Get weighted mean and covariance
        if np.sum(self.pf.weights) == 0:
            self.is_bad_weights = True
            print("Warning: All particle weights are zero.")
            return
        self.weighted_mean, self.covariance = self.pf.estimate(
                            self.pf.particles[-1], self.pf.weights)

class SimpleGNNAssociator:
    """
    Simplified Global Nearest Neighbor (GNN) data association for particle filters.
    This is a simpler version compared to the Stone Soup implementation.
    """
    def __init__(self, max_association_distance=3.0):
        self.max_association_distance = max_association_distance

    def associate(self, tracks, detections):
        """
        Perform data association between tracks and detections.
        
        Args:
            tracks: List of ParticleFilterTrack objects
            detections: List of detection arrays [x, y]
            
        Returns:
            associations: List of (track_idx, detection_idx) tuples
            unassociated_tracks: List of track indices without associations
            unassociated_detections: List of detection indices without associations
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        # Compute cost matrix (distances)
        cost_matrix = np.full((len(tracks), len(detections)), np.inf)
        
        for i, track in enumerate(tracks):
            track_state = track.weighted_mean[:2]  # Get position
            for j, detection in enumerate(detections):
                distance = np.linalg.norm(track_state - detection[:2])
                if distance <= self.max_association_distance:
                    cost_matrix[i, j] = distance
        
        # Simple greedy assignment (can be improved with Hungarian algorithm)
        associations = []
        used_tracks = set()
        used_detections = set()
        
        # Find all valid associations and sort by cost
        valid_associations = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if cost_matrix[i, j] < np.inf:
                    valid_associations.append((cost_matrix[i, j], i, j))
        
        valid_associations.sort()  # Sort by cost
        
        # Greedily assign based on lowest cost
        for cost, track_idx, det_idx in valid_associations:
            if track_idx not in used_tracks and det_idx not in used_detections:
                associations.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        unassociated_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unassociated_detections = [i for i in range(len(detections)) if i not in used_detections]
        
        return associations, unassociated_tracks, unassociated_detections


class ParticleFilterGNNAssociator:
    """
    Multi-target tracker using particle filters with GNN data association.
    This is the particle filter equivalent of KalmanFilterGNNAssociator.
    """
    def __init__(self, num_particles=200, prediction_method="Velocity", 
                 max_association_distance=3.0, max_missed_detections=35,
                 min_hits_for_confirmation=1, model_path=None, drone_height=2.0,
                 initial_time=0.0):
        """
        Initialize the multi-target particle filter tracker.
        
        Args:
            num_particles: Number of particles per track
            prediction_method: Prediction method for particle filters
            max_association_distance: Maximum distance for data association
            max_missed_detections: Maximum missed detections before track deletion
            min_hits_for_confirmation: Minimum hits before track is confirmed
        """
        self.num_particles = num_particles
        self.prediction_method = prediction_method
        self.model_path = model_path
        self.drone_height = drone_height
        # Data association
        self.associator = SimpleGNNAssociator(max_association_distance)
        
        # Track management
        self.tracks = []
        self.next_track_id = 0
        self.max_missed_detections = max_missed_detections
        self.min_hits_for_confirmation = min_hits_for_confirmation
        
        # Timing
        self.current_time = initial_time

    def update_time(self, time_step):
        """Update the current time for tracking."""
        self.current_time = time_step

    def predict_tracks(self, dt=0.1, **kwargs):
        """Perform prediction step for all tracks"""
        for track in self.tracks:
            track.predict(dt, **kwargs)

    def process_detections(self, detections, confidences=None, current_time=None):
        """
        Process new detections and update tracks.
        
        Args:
            detections: List of [x, y] positions or numpy array of shape (N, 2)
            current_time: Current simulation time 
            confidences: List of confidence values (0-1) for each detection, optional
        """
        # Convert detections to numpy array if needed
        if isinstance(detections, list) and len(detections) > 0:
            detections = np.array(detections)
        elif len(detections) == 0:
            detections = np.array([]).reshape(0, 2)
            
        # Use default confidence if not provided
        if confidences is None:
            confidences = [1.0] * len(detections)
        
        # Data association
        associations, unassociated_tracks, unassociated_detections = \
            self.associator.associate(self.tracks, detections)
        
        # Update associated tracks
        for track_idx, det_idx in associations:
            self.tracks[track_idx].pf.is_update = True
            self.tracks[track_idx].pf.resample_index = np.arange(self.num_particles)
            self.tracks[track_idx].pf.pf_loop(noisy_measurement=detections[det_idx], t=self.current_time)
            self.tracks[track_idx].pf.is_update = False
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].missed_detections = 0
            self.tracks[track_idx]._update_state_estimates()

        # loop without a detection for unassociated tracks
        for track_idx in unassociated_tracks:
            self.tracks[track_idx].pf.resample_index = np.where(
                np.logical_and(
                    self.is_in_FOV(self.tracks[track_idx].pf.particles[-1], self.FOV),
                    ~self.in_occlusion(self.tracks[track_idx].pf.particles[-1, :, :2]),
                )
            )[0]
            if self.tracks[track_idx].pf.t_since_last_update > 1.0:
                # Reduce weights for particles that haven't been updated
                self.tracks[track_idx].pf.weights *= 0.9
                # normalize weights
                self.tracks[track_idx].pf.weights = self.tracks[track_idx].pf.weights / np.sum(self.tracks[track_idx].pf.weights)
            self.tracks[track_idx].pf.is_update = False
            self.tracks[track_idx].pf.pf_loop(noisy_measurement=np.array([0, 0]), t=self.current_time)
            self.tracks[track_idx]._update_state_estimates()
            self.tracks[track_idx].missed_detections += 1

        # Create new tracks from unassociated detections
        for det_idx in unassociated_detections:
            self._create_new_track(detections[det_idx], confidences[det_idx])
        
        # Delete old tracks
        self._delete_old_tracks()

        self.current_time = current_time

    def _create_new_track(self, detection, confidence=1.0):
        """Create a new track from an unassociated detection"""
        new_track = ParticleFilterTrack(
            track_id=self.next_track_id,
            initial_detection=detection,
            initial_time=self.current_time,
            num_particles=self.num_particles,
            prediction_method=self.prediction_method,
            model_path=self.model_path,
            drone_height=self.drone_height
        )
        self.tracks.append(new_track)
        self.next_track_id += 1

    def _delete_old_tracks(self):
        """Delete tracks that have missed too many detections"""
        tracks_to_keep = []
        for track in self.tracks:
            # Delete if too many missed detections
            if track.missed_detections > self.max_missed_detections:
                print(f"Deleted id: {track.id} due to missed detections > {self.max_missed_detections}")
                continue
            if track.is_bad_weights:
                print(f"Deleted id: {track.id} due to bad weights")
                continue
            outbounds = track.pf.outside_bounds(track.pf.particles[-1])
            if outbounds > track.pf.N* 0.8:
                print(f"Deleted id: {track.id} due to {outbounds} particles being out of bounds")
                continue
            # Keep confirmed tracks or tentative tracks that might still be good
            tracks_to_keep.append(track)
        
        self.tracks = tracks_to_keep

    def get_confirmed_tracks(self):
        """Get only confirmed tracks (with enough hits)"""
        return [track for track in self.tracks 
                if track.hits >= self.min_hits_for_confirmation]

    def get_track_states(self, confirmed_only=True):
        """
        Get current track states as a list of state arrays.
        
        Args:
            confirmed_only: If True, only return confirmed tracks
            
        Returns:
            List of numpy arrays, each containing [x, y] for a track
        """
        tracks_to_use = self.get_confirmed_tracks() if confirmed_only else self.tracks
        return [track.weighted_mean for track in tracks_to_use]

    def get_track_covariances(self, confirmed_only=True):
        """
        Get current track covariances.
        
        Args:
            confirmed_only: If True, only return confirmed tracks
            
        Returns:
            List of numpy arrays, each containing the covariance matrix for a track
        """
        tracks_to_use = self.get_confirmed_tracks() if confirmed_only else self.tracks
        return [track.covariance for track in tracks_to_use]

    def get_track_particles(self, confirmed_only=True):
        """
        Get particles for all tracks.
        
        Args:
            confirmed_only: If True, only return confirmed tracks
            
        Returns:
            List of particle arrays for each track
        """
        tracks_to_use = self.get_confirmed_tracks() if confirmed_only else self.tracks
        return [track.pf.particles for track in tracks_to_use]

    def get_track_weights(self, confirmed_only=True):
        """
        Get particle weights for all tracks.
        
        Args:
            confirmed_only: If True, only return confirmed tracks
            
        Returns:
            List of weight arrays for each track
        """
        tracks_to_use = self.get_confirmed_tracks() if confirmed_only else self.tracks
        return [track.pf.weights for track in tracks_to_use]

    def get_entropy_estimates(self):
        """
        Compute entropy estimates for each track based on particle spread.
        
        Returns:
            List of entropy values for each track
        """
        entropies = []
        for track in self.tracks:
            # Compute entropy based on particle covariance
            particles = track.pf.particles[-1, :, :2]  # Position only
            weights = track.pf.weights
            
            # Weighted covariance
            mean = np.average(particles, weights=weights, axis=0)
            diff = particles - mean
            cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                           weights=weights, axis=0)
            
            # Entropy = 0.5 * log((2*pi*e)^n * |Sigma|)
            det_cov = np.linalg.det(cov + 1e-6 * np.eye(2))  # Add small regularization
            entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * max(det_cov, 1e-6))
            entropies.append(entropy)
        
        return entropies
