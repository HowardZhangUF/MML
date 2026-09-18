"""
Metrics for evaluating multi-target tracking performance.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import os


class FilterBenchmarkTracker:
    """Collect the paper's tracking metrics from the full simulation stack.

    Position errors are computed after Hungarian assignment between true targets
    and confirmed tracks. Unmatched targets are excluded from RMSE and reported
    separately through ``matched_target_fraction`` so track loss cannot be hidden
    during auditing. Visibility means that a measurement is available: the true
    target is inside the agent FOV and outside every occlusion region.
    """

    def __init__(self, recovery_threshold=0.3):
        self.recovery_threshold = float(recovery_threshold)
        self.all_errors = []
        self.visible_errors = []
        self.occluded_errors = []
        self.recovery_steps = []
        self.runtime_ms = []
        self.matched_targets = 0
        self.total_targets = 0
        self._previous_visibility = None
        self._pending_recovery = None
        self._has_been_visible = None

    def append(self, true_positions, tracked_positions, visible, runtime_ms):
        true_positions = np.asarray(true_positions, dtype=float).reshape(-1, 2)
        tracked_positions = np.asarray(tracked_positions, dtype=float).reshape(-1, 2)
        visible = np.asarray(visible, dtype=bool).reshape(-1)

        if len(visible) != len(true_positions):
            raise ValueError(
                "Visibility mask and true target count must match: "
                f"{len(visible)} != {len(true_positions)}"
            )

        assigned_errors = np.full(len(true_positions), np.nan, dtype=float)
        if len(true_positions) and len(tracked_positions):
            distances = cdist(true_positions, tracked_positions, metric="euclidean")
            true_idx, track_idx = linear_sum_assignment(distances)
            assigned_errors[true_idx] = distances[true_idx, track_idx]

        matched = np.isfinite(assigned_errors)
        self.all_errors.extend(assigned_errors[matched].tolist())
        self.visible_errors.extend(assigned_errors[matched & visible].tolist())
        self.occluded_errors.extend(assigned_errors[matched & ~visible].tolist())
        self.matched_targets += int(matched.sum())
        self.total_targets += int(len(true_positions))
        self.runtime_ms.append(float(runtime_ms))

        if self._previous_visibility is None:
            self._previous_visibility = visible.copy()
            self._pending_recovery = [None] * len(true_positions)
            self._has_been_visible = visible.copy()
            return assigned_errors

        if len(self._previous_visibility) != len(visible):
            raise ValueError("The number of true targets changed during a benchmark run.")

        for target_id in range(len(true_positions)):
            if (
                visible[target_id]
                and not self._previous_visibility[target_id]
                and self._has_been_visible[target_id]
            ):
                self._pending_recovery[target_id] = 0

            if visible[target_id]:
                self._has_been_visible[target_id] = True

            pending = self._pending_recovery[target_id]
            if pending is None:
                continue

            error = assigned_errors[target_id]
            if np.isfinite(error) and error < self.recovery_threshold:
                self.recovery_steps.append(float(pending))
                self._pending_recovery[target_id] = None
            else:
                self._pending_recovery[target_id] = pending + 1

        self._previous_visibility = visible.copy()
        return assigned_errors

    @staticmethod
    def _rmse(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(values ** 2)))

    def summary(
        self,
        tracker,
        run,
        seed,
        n_steps,
        initial_target_mode=None,
        initial_target_positions=None,
    ):
        row = {
            "tracker": tracker,
            "run": int(run),
            "seed": int(seed),
            "n_steps": int(n_steps),
            "protocol": "info_mml_main.py/MultiPFInfo full-system",
            "rmse_scope": "Hungarian-matched confirmed tracks",
            "runtime_scope": "guidance_filter_loop per tracking update",
            "rmse_all": self._rmse(self.all_errors),
            "rmse_visible": self._rmse(self.visible_errors),
            "rmse_occluded": self._rmse(self.occluded_errors),
            "recovery_steps": (
                float(np.mean(self.recovery_steps))
                if self.recovery_steps
                else float("nan")
            ),
            "ms_per_step": (
                float(np.mean(self.runtime_ms))
                if self.runtime_ms
                else float("nan")
            ),
            "matched_target_fraction": (
                float(self.matched_targets / self.total_targets)
                if self.total_targets
                else float("nan")
            ),
            "n_matched_targets": int(self.matched_targets),
            "n_target_samples": int(self.total_targets),
            "n_recoveries": int(len(self.recovery_steps)),
        }

        if initial_target_mode is not None:
            row["initial_target_mode"] = str(initial_target_mode)
        if initial_target_positions is not None:
            positions = np.asarray(initial_target_positions, dtype=float).reshape(-1, 2)
            row["initial_target_positions_m"] = ";".join(
                f"{position[0]:.6f},{position[1]:.6f}"
                for position in positions
            )
        return row

    def save_run_summary(
        self,
        directory,
        tracker,
        run,
        seed,
        n_steps,
        initial_target_mode=None,
        initial_target_positions=None,
    ):
        os.makedirs(directory, exist_ok=True)
        row = self.summary(
            tracker,
            run,
            seed,
            n_steps,
            initial_target_mode=initial_target_mode,
            initial_target_positions=initial_target_positions,
        )
        filename = f"{tracker}_run{int(run):03d}_benchmark.csv"
        path = os.path.join(directory, filename)
        pd.DataFrame([row]).to_csv(path, index=False)
        print(f"Benchmark summary saved to: {path}")
        return path

def wasserstein_distance_2d(true_positions, tracked_positions):
    """
    Compute the 2D Wasserstein distance between true target positions and tracked positions.
    
    For 2D points, we use the Hungarian algorithm to find the optimal assignment between
    true and tracked positions, then compute the average distance.
    
    Parameters:
    -----------
    true_positions : np.ndarray
        Array of shape (N, 2) containing true target positions
    tracked_positions : np.ndarray
        Array of shape (M, 2) containing tracked target positions
        
    Returns:
    --------
    float
        Wasserstein distance between the two sets of positions
    """
    if len(true_positions) == 0 and len(tracked_positions) == 0:
        return 0.0
    
    if len(true_positions) == 0:
        # No true targets, penalize for false positives
        return np.mean(np.linalg.norm(tracked_positions, axis=1))
    
    if len(tracked_positions) == 0:
        # No tracked targets, penalize for missed targets
        return np.mean(np.linalg.norm(true_positions, axis=1))
    
    # Compute pairwise distances
    distances = cdist(true_positions, tracked_positions, metric='euclidean')
    
    # Use Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Get the cost of optimal assignment
    assigned_cost = distances[row_indices, col_indices].sum()
    
    # Handle unmatched targets/tracks
    n_true = len(true_positions)
    n_tracked = len(tracked_positions)
    
    if n_true > n_tracked:
        # More true targets than tracks - penalize missed detections
        # Add penalty for unassigned true targets (assume they're at distance = max existing distance)
        max_distance = np.max(distances) if distances.size > 0 else 1.0
        unassigned_penalty = (n_true - n_tracked) * max_distance
        total_cost = assigned_cost + unassigned_penalty
        return total_cost / n_true
    elif n_tracked > n_true:
        # More tracks than true targets - penalize false positives
        # Add penalty for unassigned tracks
        max_distance = np.max(distances) if distances.size > 0 else 1.0
        unassigned_penalty = (n_tracked - n_true) * max_distance
        total_cost = assigned_cost + unassigned_penalty
        return total_cost / n_tracked
    else:
        # Equal numbers - just return average assigned distance
        return assigned_cost / n_true


def optimal_subpattern_assignment_distance(
    true_positions,
    tracked_positions,
    c=1.5,
    p=2,
):
    """
    Compute the Optimal SubPattern Assignment (OSPA) distance.
    
    This is a popular metric in multi-target tracking that handles cardinality differences
    and provides a principled way to penalize both localization errors and cardinality errors.
    
    Parameters:
    -----------
    true_positions : np.ndarray
        Array of shape (N, 2) containing true target positions
    tracked_positions : np.ndarray
        Array of shape (M, 2) containing tracked target positions
    c : float, optional
        Cut-off parameter (default: 1.5)
    p : int, optional
        Order parameter (default: 2)
        
    Returns:
    --------
    float
        OSPA distance
    """
    c = float(c)
    p = float(p)
    if c <= 0 or p < 1:
        raise ValueError("OSPA requires c > 0 and p >= 1.")

    n_true = len(true_positions)
    n_tracked = len(tracked_positions)
    
    if n_true == 0 and n_tracked == 0:
        return 0.0
    
    if n_true == 0:
        return c
    
    if n_tracked == 0:
        return c
    
    # Standard OSPA: assignment costs are capped and raised to order p;
    # each unmatched object contributes c**p before normalization by max(n, m).
    distances = cdist(true_positions, tracked_positions, metric='euclidean')
    costs = np.minimum(distances, c) ** p
    row_indices, col_indices = linear_sum_assignment(costs)
    localization_cost = costs[row_indices, col_indices].sum()
    cardinality_cost = (c ** p) * abs(n_true - n_tracked)
    return ((localization_cost + cardinality_cost) / max(n_true, n_tracked)) ** (1 / p)


def determinant_of_covariance(covs):
    """
    Compute the average of the determinant of the covariance matrices of tracked targets.

    Parameters:
    -----------
    covs : np.ndarray
        Array of shape (M, 2, 2) containing tracked target covariances

    Returns:
    --------
    float
        Average of the determinants of the covariance matrices
    """
    if len(covs) == 0:
        return None

    dets = []
    for covariance in covs:
        covariance = np.asarray(covariance)
        if covariance.ndim != 2 or covariance.shape[0] < 2 or covariance.shape[1] < 2:
            continue
        position_covariance = covariance[:2, :2]
        dets.append(np.linalg.det(position_covariance + 1e-6 * np.eye(2)))
    if not dets:
        return None
    return np.mean(dets)

class MetricsTracker:
    """
    A class to track and analyze multi-target tracking performance metrics over time.
    """
    
    def __init__(self):
        """Initialize the metrics tracker with empty storage."""
        self.metrics_history = []
        
        # Define available metrics based on existing functions
        self.available_metrics = []
        self.metric_functions = {}
        
        if 'wasserstein_distance_2d' in globals():
            self.available_metrics.append('wasserstein_2d')
            self.metric_functions['wasserstein_2d'] = wasserstein_distance_2d
        
        if 'optimal_subpattern_assignment_distance' in globals():
            self.available_metrics.append('ospa')
            self.metric_functions['ospa'] = optimal_subpattern_assignment_distance

        if 'determinant_of_covariance' in globals():
            self.available_metrics.append('determinant_of_covariance')
            self.metric_functions['determinant_of_covariance'] = determinant_of_covariance

        # Always available metrics (simple calculations)
        self.available_metrics.extend(['cardinality_error', 'rmse'])
        
        # Initialize cumulative metrics dictionary with dynamic keys
        self.cumulative_metrics = {
            'step': [],
            'time': []
        }
        for metric in self.available_metrics:
            self.cumulative_metrics[metric] = []
        
        print(f"MetricsTracker initialized with metrics: {self.available_metrics}")
    
    def get_available_metrics(self):
        """
        Get list of available metrics.
        
        Returns:
        --------
        list
            List of available metric names
        """
        return self.available_metrics.copy()
    
    def _compute_metrics(self, true_positions, tracked_positions, covs):
        """
        Compute all available metrics for the given positions.
        
        Parameters:
        -----------
        true_positions : np.ndarray
            Array of shape (N, 2) containing true target positions
        tracked_positions : np.ndarray
            Array of shape (M, 2) containing tracked target positions
        covs : np.ndarray
            Array of shape (M, 2, 2) containing tracked target covariances
        Returns:
        --------
        dict
            Dictionary containing computed metrics
        """
        true_positions = np.array(true_positions).reshape(-1, 2)
        tracked_positions = np.array(tracked_positions).reshape(-1, 2)
        
        metrics = {}
        
        # Compute metrics using available functions
        for metric_name, metric_func in self.metric_functions.items():
            try:
                if metric_name == 'determinant_of_covariance':
                    metrics[metric_name] = metric_func(covs)
                else:
                    metrics[metric_name] = metric_func(true_positions, tracked_positions)
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = None
        
        # Compute cardinality error (always available)
        if 'cardinality_error' in self.available_metrics:
            metrics['cardinality_error'] = abs(len(true_positions) - len(tracked_positions))
        
        # Compute RMSE (always available, but only meaningful for equal cardinalities)
        if 'rmse' in self.available_metrics:
            if len(true_positions) == len(tracked_positions) and len(true_positions) > 0:
                distances = cdist(true_positions, tracked_positions, metric='euclidean')
                row_indices, col_indices = linear_sum_assignment(distances)
                rmse = np.sqrt(np.mean(distances[row_indices, col_indices] ** 2))
                metrics['rmse'] = rmse
            else:
                metrics['rmse'] = None
        
        return metrics
    
    def append_metrics(self, true_positions, tracked_positions, covs, step=None, time=None):
        """
        Compute and store metrics for the current time step.
        
        Parameters:
        -----------
        true_positions : np.ndarray
            Array of shape (N, 2) containing true target positions
        tracked_positions : np.ndarray
            Array of shape (M, 2) containing tracked target positions
        covs : np.ndarray
            Array of shape (M, 2, 2) containing tracked target covariances
        step : int, optional
            Current simulation step
        time : float, optional
            Current simulation time
        
        Returns:
        --------
        dict
            Dictionary containing computed metrics for current step
        """
        # Compute current metrics using internal method
        current_metrics = self._compute_metrics(true_positions, tracked_positions, covs)
        self.metrics_history.append(current_metrics)
            
        # Store in cumulative dictionary
        self.cumulative_metrics['step'].append(step if step is not None else len(self.metrics_history) - 1)
        self.cumulative_metrics['time'].append(time if time is not None else 0.0)
        for key in self.available_metrics:
            self.cumulative_metrics[key].append(current_metrics[key])
    
    def print_step_metrics(self):
        """
        Print metrics for the current step.
        
        Parameters:
        -----------
        step : int
            Current simulation step
        current_metrics : dict
            Metrics dictionary for current step
        """
        if len(self.metrics_history) == 0:
            print("No metrics to display.")
            return
        print(f"Step {self.cumulative_metrics['step'][-1]} Metrics: ", end="")
        for metric in self.available_metrics:
            value = self.metrics_history[-1].get(metric)
            if isinstance(value, float):
                if value >= 1e-3:
                    print(f"{metric}={value:.2f}", end=", ")
                else:
                    print(f"{metric}={value:.2e}", end=", ")
            else:
                print(f"{metric}={value}", end=", ")
        print()  # New line at the end
    
    def print_summary(self):
        """Print comprehensive summary statistics of all collected metrics."""
        print("\n" + "="*50)
        print("TRACKING PERFORMANCE SUMMARY")
        print("="*50)
        
        if len(self.metrics_history) == 0:
            print("No metrics were collected during simulation.")
            return
        
        print(f"Available metrics: {', '.join(self.available_metrics)}")
        print(f"Total tracking steps: {len(self.metrics_history)}")
        
        # Print statistics for each available metric
        for metric in self.available_metrics:
            if metric in self.cumulative_metrics and len(self.cumulative_metrics[metric]) > 0:
                values = self.cumulative_metrics[metric]
                
                if metric == 'rmse' or metric == 'determinant_of_covariance':
                    # Special handling for RMSE (exclude None values)
                    valid_values = [v for v in values if v is not None]
                    if len(valid_values) > 0:
                        print(f"Average {metric} (when cardinalities match): {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")
                        print(f"{metric} available for {len(valid_values)}/{len(values)} steps ({100*len(valid_values)/len(values):.1f}%)")
                    else:
                        print(f"{metric}: No steps with matching cardinalities")
                else:
                    # Standard metrics
                    print(f"Average {metric.replace('_', ' ').title()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Min/Max statistics for numerical metrics
        print(f"\nMin/Max Values:")
        for metric in self.available_metrics:
            if metric in self.cumulative_metrics and len(self.cumulative_metrics[metric]) > 0:
                values = self.cumulative_metrics[metric]
                if metric == 'rmse' or metric == 'determinant_of_covariance':
                    valid_values = [v for v in values if v is not None]
                    if len(valid_values) > 0:
                        print(f"  {metric}: Min={np.min(valid_values):.4f}, Max={np.max(valid_values):.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: Min={np.min(values):.4f}, Max={np.max(values):.4f}")
        
        # Cardinality analysis (if available)
        if 'cardinality_error' in self.cumulative_metrics:
            cardinality_errors = np.array(self.cumulative_metrics['cardinality_error'])
            perfect_tracking_steps = np.sum(cardinality_errors == 0)
            print(f"\nPerfect cardinality tracking: {perfect_tracking_steps}/{len(cardinality_errors)} steps ({100*perfect_tracking_steps/len(cardinality_errors):.1f}%)")
    
    def save_to_csv(self, prefix_name="", directory="tracking_results"):
        """
        Save all metrics data to a CSV file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the CSV file. If None, generates timestamp-based name.
        directory : str, optional
            Directory to save the file. Created if it doesn't exist.
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_name = prefix_name + "_" if prefix_name else prefix_name
        filename = prefix_name + f"tracking_metrics_{timestamp}.csv"
        
        filepath = os.path.join(directory, filename)
        
        # Create DataFrame from cumulative metrics
        df = pd.DataFrame(self.cumulative_metrics)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        print(f"Metrics saved to: {filepath}")
        print(f"Data shape: {df.shape}")
        return filepath
    
    def reset(self):
        """Reset all stored metrics."""
        self.metrics_history = []
        for key in self.cumulative_metrics:
            self.cumulative_metrics[key] = []


# =============================================================================
# Occlusion Recovery Tracker
# =============================================================================

class OcclusionRecoveryTracker:
    """
    Tracks whether the motion model correctly predicts the exit direction when a
    target passes through an occlusion zone.

    Two scores are recorded per event:

    Modal-side score (primary)
    --------------------------
    The majority of the trajectory fan samples vote for an exit side (above/below
    the occlusion centre y-coordinate).  Score = 1 if that majority matches the
    actual exit, 0 otherwise.  This is comparable between CFM and Transformer
    because both produce a single binary decision.

    Probability score (secondary / calibration)
    --------------------------------------------
    The fraction of fan samples that voted for the *correct* exit side.  For the
    Transformer this is always 0.0 or 1.0 (deterministic output + cosmetic
    noise).  For CFM it is a genuine probability, which lets you test whether
    CFM's uncertainty is calibrated — i.e. does "70 % of samples say up" actually
    translate to a ~70 % correct rate across many events?

    Usage
    -----
    Call ``record_entry`` the step the target enters occlusion (pass the
    trajectory fan at that moment).  Call ``record_exit`` the step it re-appears.
    Call ``print_summary`` / ``save_to_csv`` at the end of the run.
    """

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.events: list[dict] = []          # one dict per completed event
        self._pending: dict[int, dict] = {}   # target_id → open event

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        target_id: int,
        step: int,
        occ_center_y: float,
        fan: np.ndarray | None,
        fan_weights: np.ndarray | None,
        last_known_pos: np.ndarray,
        last_known_vel: np.ndarray,
    ) -> None:
        """
        Call when target_id transitions visible → occluded.

        Parameters
        ----------
        occ_center_y:
            The y-coordinate of the occlusion zone centre the target entered.
            Used as the split boundary: fan samples with final y > occ_center_y
            are "above", rest are "below".
        fan:
            Shape (n_samples, n_steps, 2) in metres, or None if unavailable.
        fan_weights:
            Shape (n_samples,) normalised weights, or None.
        last_known_pos / last_known_vel:
            Position and velocity at the last visible frame — stored so a
            velocity-baseline fallback can be computed even when no fan exists.
        """
        predicted_side, prob_above = self._score_fan(
            fan, fan_weights, occ_center_y, last_known_pos, last_known_vel
        )

        self._pending[target_id] = {
            "target_id":       target_id,
            "entry_step":      step,
            "occ_center_y":    occ_center_y,
            "predicted_side":  predicted_side,   # "above" | "below" | None
            "prob_above":      prob_above,        # float in [0, 1] or None
            "last_known_pos":  last_known_pos.copy(),
            "last_known_vel":  last_known_vel.copy(),
        }

    def record_exit(
        self,
        target_id: int,
        step: int,
        exit_pos: np.ndarray,
    ) -> dict | None:
        """
        Call when target_id transitions occluded → visible.

        Returns the completed event dict (or None if no matching entry exists).
        """
        if target_id not in self._pending:
            return None

        ev = self._pending.pop(target_id)
        occ_center_y  = ev["occ_center_y"]
        predicted_side = ev["predicted_side"]
        prob_above     = ev["prob_above"]

        actual_side = "above" if exit_pos[1] > occ_center_y else "below"

        # Modal-side binary score.
        if predicted_side is None:
            modal_correct = None
        else:
            modal_correct = int(predicted_side == actual_side)

        # Probability score: fraction of mass on the correct side.
        if prob_above is None:
            prob_correct = None
        elif actual_side == "above":
            prob_correct = prob_above
        else:
            prob_correct = 1.0 - prob_above

        duration_steps = step - ev["entry_step"]

        event = {
            "target_id":       target_id,
            "entry_step":      ev["entry_step"],
            "exit_step":       step,
            "duration_steps":  duration_steps,
            "occ_center_y":    occ_center_y,
            "predicted_side":  predicted_side,
            "actual_side":     actual_side,
            "modal_correct":   modal_correct,   # 1 / 0 / None
            "prob_correct":    prob_correct,     # float in [0,1] / None
        }
        self.events.append(event)
        return event

    def discard_pending(self, target_id: int) -> None:
        """Drop an open entry event without scoring (e.g. run ended mid-occlusion)."""
        self._pending.pop(target_id, None)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def modal_recovery_rate(self) -> float | None:
        """Fraction of events where modal-side prediction was correct."""
        scored = [e["modal_correct"] for e in self.events if e["modal_correct"] is not None]
        if not scored:
            return None
        return float(np.mean(scored))

    def mean_prob_correct(self) -> float | None:
        """
        Mean probability score across events (CFM calibration metric).
        A perfectly calibrated model returns 1.0 here if it always puts all
        mass on the correct side, 0.5 for chance.
        """
        scored = [e["prob_correct"] for e in self.events if e["prob_correct"] is not None]
        if not scored:
            return None
        return float(np.mean(scored))

    def calibration_data(self, n_bins: int = 5) -> dict:
        """
        Bin events by predicted probability and return observed accuracy per bin.
        Useful for reliability diagrams.

        Returns dict with keys 'bin_centers', 'bin_accuracy', 'bin_counts'.
        """
        prob_correct = np.array(
            [e["prob_correct"] for e in self.events if e["prob_correct"] is not None],
            dtype=np.float32,
        )
        modal_correct = np.array(
            [e["modal_correct"] for e in self.events
             if e["modal_correct"] is not None and e["prob_correct"] is not None],
            dtype=np.float32,
        )
        if len(prob_correct) == 0:
            return {"bin_centers": [], "bin_accuracy": [], "bin_counts": []}

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers, accuracies, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (prob_correct >= lo) & (prob_correct < hi)
            cnt = int(mask.sum())
            centers.append(float((lo + hi) / 2))
            accuracies.append(float(modal_correct[mask].mean()) if cnt > 0 else float("nan"))
            counts.append(cnt)

        return {
            "bin_centers":   centers,
            "bin_accuracy":  accuracies,
            "bin_counts":    counts,
        }

    def print_summary(self) -> None:
        n = len(self.events)
        print(f"\n{'=' * 60}")
        print(f"OCCLUSION RECOVERY SUMMARY  [{self.model_name}]")
        print(f"{'=' * 60}")
        print(f"Total occlusion events scored : {n}")

        if n == 0:
            print("No events to report.")
            return

        mrr = self.modal_recovery_rate()
        mpc = self.mean_prob_correct()

        if mrr is not None:
            print(f"Modal-side recovery rate      : {mrr * 100:.1f}%  ({int(round(mrr * n))}/{n})")
        if mpc is not None:
            print(f"Mean probability score        : {mpc:.3f}  (chance = 0.500)")

        durations = [e["duration_steps"] for e in self.events]
        print(f"Occlusion duration  mean/max  : {np.mean(durations):.1f} / {np.max(durations)} steps")

        cal = self.calibration_data()
        if any(c > 0 for c in cal["bin_counts"]):
            print("\nCalibration (predicted prob → observed accuracy):")
            for ctr, acc, cnt in zip(cal["bin_centers"], cal["bin_accuracy"], cal["bin_counts"]):
                if cnt > 0:
                    bar = "#" * cnt
                    acc_str = f"{acc * 100:.0f}%" if not np.isnan(acc) else "  -"
                    print(f"  [{ctr:.2f}]  {acc_str:>5}  n={cnt}  {bar}")
        print("=" * 60)

    def save_to_csv(self, prefix_name: str = "", directory: str = "tracking_results") -> str:
        """Save per-event data to CSV alongside the main metrics file."""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (prefix_name + "_") if prefix_name else ""
        filename = f"{prefix}occlusion_recovery_{self.model_name}_{timestamp}.csv"
        filepath = os.path.join(directory, filename)

        rows = []
        for e in self.events:
            rows.append({
                "target_id":      e["target_id"],
                "entry_step":     e["entry_step"],
                "exit_step":      e["exit_step"],
                "duration_steps": e["duration_steps"],
                "occ_center_y":   e["occ_center_y"],
                "predicted_side": e["predicted_side"],
                "actual_side":    e["actual_side"],
                "modal_correct":  e["modal_correct"],
                "prob_correct":   e["prob_correct"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Occlusion recovery data saved to: {filepath}  ({len(rows)} events)")
        return filepath

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_fan(
        self,
        fan: np.ndarray | None,
        fan_weights: np.ndarray | None,
        occ_center_y: float,
        last_known_pos: np.ndarray,
        last_known_vel: np.ndarray,
    ) -> tuple[str | None, float | None]:
        """
        Returns (predicted_side, prob_above) from the trajectory fan.

        Falls back to a single-step velocity extrapolation when no fan is
        available (e.g. MultiPFVel baseline).
        """
        if fan is not None and fan.ndim == 3 and fan.shape[0] > 0 and fan.shape[1] > 0:
            # Use the final predicted position of each sample.
            final_y = fan[:, -1, 1]   # shape (n_samples,)

            if fan_weights is not None and len(fan_weights) == len(final_y):
                weights = fan_weights.astype(np.float64)
                w_sum = weights.sum()
                if w_sum > 1e-12:
                    weights /= w_sum
                else:
                    weights = np.ones(len(final_y)) / len(final_y)
            else:
                weights = np.ones(len(final_y)) / len(final_y)

            # Weighted fraction of samples predicting exit above occ_center_y.
            prob_above = float(weights[final_y > occ_center_y].sum())
            predicted_side = "above" if prob_above >= 0.5 else "below"
            return predicted_side, prob_above

        # Velocity fallback: extrapolate one step from last known state.
        if last_known_vel is not None and np.any(last_known_vel != 0):
            predicted_y = last_known_pos[1] + last_known_vel[1]
            predicted_side = "above" if predicted_y > occ_center_y else "below"
            # Binary — no fan available, so probability is 0 or 1.
            prob_above = 1.0 if predicted_side == "above" else 0.0
            return predicted_side, prob_above

        return None, None


# # Example usage and testing
# if __name__ == "__main__":
#     # Test with some example data
#     tracker = MetricsTracker()
    
#     # Simulate a few tracking steps
#     true_targets = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     tracked_targets = np.array([[1.1, 2.1], [2.9, 4.1], [5.2, 5.8]])
    
#     print("Testing MetricsTracker:")
#     print("=" * 30)
    
#     # Add metrics for a few steps
#     for step in range(3):
#         # Add some noise to simulate tracking evolution
#         noise = np.random.normal(0, 0.1, tracked_targets.shape)
#         noisy_tracked = tracked_targets + noise
        
#         current_metrics = tracker.append_metrics(true_targets, noisy_tracked, step=step, time=step*0.1)
#         tracker.print_step_metrics(step, current_metrics)
    
#     # Print summary
#     tracker.print_summary()
    
#     # Save to CSV
#     filepath = tracker.save_to_csv("test_metrics.csv")

if __name__ == "__main__":
    """ reader for the csv files. Read indiviual csv and plot the results along with their statistics """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Set the name of the input CSV directory
    folder_path = "./tracking_results/"
    filenames = os.listdir(folder_path)
    print("filenames: ", filenames)

    for filename in filenames:
        if filename.endswith(".csv"):
            print(f"Reading file: {filename}")
            file_df = pd.read_csv(folder_path + filename, low_memory=False)
            print(file_df.describe())
            # Plotting example for 'wasserstein_2d' if it exists
            plt.figure()
            if 'wasserstein_2d' in file_df.columns:
                plt.plot(file_df['step'], file_df['wasserstein_2d'], label='Wasserstein 2D')
            if 'ospa' in file_df.columns:
                plt.plot(file_df['step'], file_df['ospa'], label='OSPA Distance', color='orange')
            if 'cardinality_error' in file_df.columns:
                plt.plot(file_df['step'], file_df['cardinality_error'], label='Cardinality Error', color='green')
            if 'rmse' in file_df.columns:
                plt.plot(file_df['step'], file_df['rmse'], label='RMSE', color='red')
            if 'determinant_of_covariance' in file_df.columns:
                plt.plot(file_df['step'], 10*file_df['determinant_of_covariance'], label='Determinant of Covariance', color='purple')
            plt.xlabel('Step')
            plt.ylabel('Metric Value')
            plt.title(f'{filename}')
            plt.legend()
            plt.grid()
            plt.show()
