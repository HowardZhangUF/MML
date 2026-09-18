"""Run info_mml_main's MML guidance and display a Stone Soup HTML replay.

Examples (from the directory containing this file):
    python example_mtt_eot_visualization.py --num_steps 400 --no-show
    python example_mtt_eot_visualization.py --prediction_method MultiPFVel --no_fan
    python example_mtt_eot_visualization.py --prediction_method PF+CFM \
        --guidance_mode MultiPFInfo --num_steps 400 --benchmark_metrics

All tracking, guidance, model, initialization and metric options come from
scripts.info_mml_main.build_parser(). This file records run_simulation() without
running a second filter or selecting actions. Stone Soup is used to represent
and plot the MML estimates, including covariance, with stable track IDs.

The camera FOV, occlusion zones and target measurements are those of the MML
runner. The former synthetic extended returns and centroid-follow controller
are removed. --steps is an alias for --num_steps. Old demo options --fov-side,
--agent-speed and --occlusion no longer apply; configure the MML runner instead.

The simulation runs headlessly first, then saves an HTML replay plus an NPZ of
states/actions. Pygame/Pymunk remains the movement backend. Learned filters and
PPO modes require checkpoints compatible with the current Guidance observation
schema. The runner's default older PPO checkpoint expects four target features,
while current attention guidance supplies five; this existing mismatch also
affects the original runner. This adapter does not change policy observations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


def parse_args(argv=None):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    here = Path(__file__).resolve().parent
    candidates = [here / "mml_python_sim", here.parent, Path.home() / "MML/mml_python_sim"]
    default_root = next((p for p in candidates if (p / "scripts/info_mml_main.py").is_file()),
                        candidates[-1])
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--project-root", type=Path, default=default_root)
    preliminary, _ = bootstrap.parse_known_args(argv)
    root = preliminary.project_root.expanduser().resolve()
    if not (root / "scripts/info_mml_main.py").is_file():
        bootstrap.error(f"Missing scripts/info_mml_main.py under {root}")
    sys.path.insert(0, str(root))
    from scripts import info_mml_main as runner

    parser = runner.build_parser()
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--steps", dest="num_steps", type=int, default=argparse.SUPPRESS,
                        help="Alias for --num_steps")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("mml_mtt_eot.html"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-particles", action="store_true",
                        help="Omit particle overlays to reduce HTML size")
    args = parser.parse_args(argv)
    args.project_root = root
    args.output = args.output.expanduser().resolve()
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args.headless = True
    if args.num_steps is None:
        args.num_steps = int(runner.load_config()["number_sim_steps"])
    if args.num_steps < 2 or args.frame_stride < 1:
        parser.error("HTML replay requires num_steps >= 2 and frame-stride >= 1")
    for name in ["model_path", "ppo_model_path"]:
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    from scripts.simulation_runner import SimulationRunner
    args.simulation_runner = SimulationRunner
    args.runner = runner
    return args


@contextmanager
def project_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def particle_points(clouds):
    """Use latest particle slice; deterministic thinning never consumes MML RNG."""
    points = []
    for cloud in clouds:
        array = np.asarray(cloud)
        if array.ndim >= 3:
            array = array[-1]
        if array.size:
            array = array.reshape(-1, array.shape[-1])[:, :2]
            points.extend(array[::max(1, int(np.ceil(len(array) / 200)))])
    return np.asarray(points, dtype=float).reshape(-1, 2).copy()


class MMLRecorder:
    def __init__(self, args):
        self.args = args
        self.start = datetime(2026, 1, 1)
        self.truths, self.tracks, self.frames = [], {}, []
        self.states, self.actions = [], []
        self.environment = None

    def __call__(self, *, step, t, state, action, detections, guidance, environment,
                 filter_updated, goal, particles, future_particles, trajectory_fan,
                 trajectory_weights, last_frame):
        self.environment = environment
        timestamp = self.start + timedelta(seconds=float(t))
        self.states.append(state.copy())
        command = np.asarray(action, dtype=float).ravel()
        self.actions.append(np.r_[command[:2], command[2] if len(command) > 2 else 0.0])
        if not self.truths:
            self.truths = [GroundTruthPath() for _ in state[1:]]
        for truth, row in zip(self.truths, state[1:]):
            truth.append(GroundTruthState(row[:2].copy(), timestamp=timestamp))

        # Adapt MML estimates for plotting; no Stone Soup filtering is performed.
        estimates = []
        if self.args.prediction_method in self.args.runner.MULTI_PF_METHODS:
            for track in guidance.multi_filter.get_confirmed_tracks():
                estimates.append((f"pf:{track.id}", track.weighted_mean[:2],
                                  track.covariance[:2, :2]))
        elif self.args.prediction_method == "MultiKF":
            for track in guidance.multi_filter.tracks:
                estimates.append((f"kf:{track.id}", track.state_vector[[0, 2], 0],
                                  track.covar[np.ix_([0, 2], [0, 2])]))
        else:
            estimates.append(("single", guidance.filter.weighted_mean[:2],
                              guidance.filter.var[:2, :2]))
        if filter_updated:
            for track_id, mean, covariance in estimates:
                if track_id not in self.tracks:
                    self.tracks[track_id] = Track(id=track_id)
                self.tracks[track_id].append(GaussianState(
                    np.asarray(mean).copy(), np.asarray(covariance).copy(), timestamp=timestamp))

        if step % self.args.frame_stride and not last_frame:
            return
        # Odd physics frames reuse the command but contain no new measurement.
        scan = {Detection(np.asarray(p).copy(), timestamp=timestamp)
                for p in detections} if filter_updated else set()
        self.frames.append(dict(
            timestamp=timestamp, t=t, agent=state[0, :2].copy(), scan=scan,
            goal=None if goal is None else np.asarray(goal)[:2].copy(),
            action=self.actions[-1].copy(), active_tracks=len(estimates),
            fov=np.asarray(guidance.FOV).copy(),
            particles=particle_points(particles) if not self.args.no_particles else [],
            future_particles=particle_points(future_particles) if not self.args.no_particles else [],
            fan=None if trajectory_fan is None else np.asarray(trajectory_fan).copy(),
            weights=None if trajectory_weights is None else np.asarray(trajectory_weights).copy(),
        ))


def scatter(points, **kwargs):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return go.Scatter(x=points[:, 0].tolist(), y=points[:, 1].tolist(), **kwargs)


def build_animation(recorder):
    frames = recorder.frames
    times = [frame["timestamp"] for frame in frames]
    plotter = AnimatedPlotterly(times, tail_length=1.0,
                               sim_duration=max(0.001, frames[-1]["t"] - frames[0]["t"]))
    plotter.plot_ground_truths(recorder.truths, [0, 1], resize=False)
    if recorder.tracks:
        plotter.plot_tracks(list(recorder.tracks.values()), [0, 1],
                            uncertainty=True, resize=False, label="MML tracks")
    plotter.time_window = timedelta(0)
    scans = [frame["scan"] for frame in frames]
    if any(scans):
        plotter.plot_measurements(scans, [0, 1], convert_measurements=False,
                                  show_clutter=False, resize=False, label="MML detections",
                                  marker=dict(color="red", symbol="cross", size=9))
    fig = plotter.fig
    names = ["Agent path", "Agent", "Square FOV", "Guidance goal", "Action (1 s)",
             "Particles", "Future particles", "Trajectory fan", "Weighted future path"]
    base = len(fig.data)
    for name in names:
        fig.add_trace(go.Scatter(x=[], y=[], name=name))
    for k, (plot_frame, frame) in enumerate(zip(fig.frames, frames)):
        agent = frame["agent"]
        xmin, xmax, ymin, ymax = frame["fov"]
        fan = frame["fan"]
        fan_points, mean = [], []
        if fan is not None and len(fan):
            for path in fan:
                fan_points.extend(path[:, :2].tolist() + [[None, None]])
            weights = frame["weights"]
            if weights is None:
                weights = np.ones(len(fan))
            weights = weights / np.sum(weights)
            mean = np.average(fan[:, :, :2], axis=0, weights=weights)
        overlays = [
            scatter([f["agent"] for f in frames[:k + 1]], mode="lines", line=dict(color="green")),
            scatter([agent], mode="markers", marker=dict(color="green", symbol="diamond", size=12)),
            scatter([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
                    mode="lines", line=dict(color="green")),
            scatter([] if frame["goal"] is None else [agent, frame["goal"]], mode="lines",
                    line=dict(color="orange", dash="dot")),
            scatter([agent, agent + frame["action"][:2]], mode="lines",
                    line=dict(color="teal", dash="dash")),
            scatter(frame["particles"], mode="markers", marker=dict(size=3, color="gray"), opacity=0.4),
            scatter(frame["future_particles"], mode="markers", marker=dict(size=3, color="cyan"), opacity=0.3),
            scatter(fan_points, mode="lines", line=dict(width=1, color="blue"), opacity=0.15),
            scatter(mean, mode="lines", line=dict(width=3, color="magenta")),
        ]
        for name, trace in zip(names, overlays):
            trace.name = name
        plot_frame.data = list(plot_frame.data) + overlays
        plot_frame.traces = list(plot_frame.traces) + list(range(base, base + len(overlays)))
        plot_frame.layout.title.text = (
            f"{recorder.args.prediction_method} / {recorder.args.guidance_mode} | "
            f"t={frame['t']:.2f} s | {len(frame['scan'])} detections | "
            f"{frame['active_tracks']} tracks")

    env = recorder.environment
    # The runner sets these exact occlusion boxes in render_init().
    for centre, width in zip(env.occ_centers, env.occ_widths):
        x, y = env.pixels_to_meters(*centre)
        half = width / env.pixels_per_meter / 2
        fig.add_shape(type="rect", x0=x-half, x1=x+half, y0=y-half, y1=y+half,
                      fillcolor="gray", opacity=0.25, layer="below", line=dict(color="black"))
    road = [env.pixels_to_meters(node["x"], node["y"]) for node in env.goal_list]
    road_points = []
    for a, b in np.argwhere(env.base_trans_matrix > 0):
        road_points.extend([road[a], road[b], [None, None]])
    fig.add_trace(scatter(road_points, name="Road network", mode="lines",
                          line=dict(color="lightgray", width=1)))
    fig.add_shape(type="rect", x0=-3, x1=3, y0=-3, y1=3, line=dict(color="black"))
    fig.update_layout(title=fig.frames[0].layout.title, height=850,
                      xaxis=dict(title="x [m]", range=[-3.2, 3.2], autorange=False),
                      yaxis=dict(title="y [m]", range=[-3.2, 3.2], autorange=False,
                                 scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=-0.3))
    for index, data in zip(fig.frames[0].traces, fig.frames[0].data):
        fig.data[index].update(data)
    return fig


def run(args):
    recorder = MMLRecorder(args)
    result = args.simulation_runner(args, frame_callback=recorder).run()
    figure = build_animation(recorder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(args.output), include_plotlyjs=True, auto_play=False,
                      auto_open=not args.no_show)
    np.savez_compressed(args.output.with_suffix(".npz"),
                        states=np.asarray(recorder.states + [result["state"]]),
                        actions=np.asarray(recorder.actions), dt=result["dt"])
    print(f"Saved {len(figure.frames)} frames: {args.output}")
    print(f"MML metrics: {args.output_dir}")
    return dict(figure=figure, recorder=recorder, simulation=result)


if __name__ == "__main__":
    run(parse_args())
