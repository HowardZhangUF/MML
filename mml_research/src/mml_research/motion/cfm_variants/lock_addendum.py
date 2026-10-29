#!/usr/bin/env python3
"""
lock_addendum.py — lock the open-source / earlier baselines for the test split.

These methods were not part of the preregistered four-variant study, so
locked.json does not cover them. Their settings come unchanged from the exit
benchmark's validation sweep (runs 100..119, selection rule: lowest mean RMSE,
sweep_process_noise.py / sweep_enff.py), made before any of them ran on the
test episodes 1000..1099. This step copies those settings and hashes the
checkpoints into locked_addendum.json; run.py --split test then checks the
addendum exactly like locked.json. It refuses to overwrite an existing lock.

    python -m mml_research.motion.cfm_variants.lock_addendum \
        --sweep artifacts/motion/exit_benchmark/noise_sweep/chosen_noise.txt
"""

from __future__ import annotations

import argparse
import os
import sys

from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, git_commit, load_config, sha256, write_json,
)
from mml_research.motion.cfm_variants.methods import METHODS, OPEN_SOURCE, checkpoint_files


def main():
    p = argparse.ArgumentParser(description="Lock the open-source baselines' validation-selected settings")
    p.add_argument("--sweep", required=True, help="exit benchmark chosen_noise.txt (validation runs 100..119)")
    p.add_argument("--enff-lambda", dest="enff_lambda", type=float, default=2.5e-4)
    p.add_argument("--enff-sigma-min", dest="enff_sigma_min", type=float, default=1e-2)
    p.add_argument("--enff-steps", dest="enff_steps", type=int, default=5)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    path = os.path.join(args.out_dir, "locked_addendum.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; a lock is never overwritten")
    with open(args.sweep) as f:
        chosen = dict(kv.split("=") for kv in f.read().split())
    q = {m: float(chosen[m]) for m in OPEN_SOURCE if METHODS[m][1] == "addendum"}
    files = checkpoint_files(OPEN_SOURCE, [], load_config(args.config), args.models_dir, args.out_dir)
    write_json(path, {
        "commit": git_commit(), "methods": OPEN_SOURCE,
        "selection": "exit benchmark validation sweep, runs 100..119, lowest mean RMSE (all steps)",
        "sweep_file": os.path.abspath(args.sweep), "sweep_sha256": sha256(args.sweep),
        "process_noise": q,
        "enff": {"lambda": args.enff_lambda, "sigma_min": args.enff_sigma_min, "steps": args.enff_steps},
        "checkpoints": {k: sha256(v) for k, v in files.items()},
    })
    print(f"locked {OPEN_SOURCE}: q={q} -> {path}")


if __name__ == "__main__":
    main()
