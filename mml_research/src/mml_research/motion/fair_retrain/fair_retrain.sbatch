#!/bin/bash
# Fair retraining on HiPerGator: data -> train (GPU) | calibrate -> test -> summarize (CPU burst).
# Submit from the repo root as two dependent jobs:
#   S=mml_research/src/mml_research/motion/fair_retrain/fair_retrain.sbatch
#   A=$(sbatch --parsable --qos=jshin2 --partition=hpg-turin --gpus=1 --cpus-per-task=8 --mem=64gb \
#        --time=3-00:00:00 --export=ALL,STAGES="data train" $S)
#   sbatch --dependency=afterok:$A --qos=jshin2-b --cpus-per-task=96 --mem=320gb --time=3-00:00:00 \
#        --export=ALL,STAGES="calibrate test summarize" $S
# Env overrides: STAGES=... OUT=... WORKERS=...
#SBATCH --job-name=fair_retrain
#SBATCH --account=jshin2
#SBATCH --qos=jshin2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/motion/fair_v1/logs/slurm_fair_retrain_%j.log

# /etc/profile and conda's hook are not `set -u` clean; enable strict mode after activation.
source /etc/profile
module load conda
eval "$(conda shell.bash hook)"
conda activate zhz14
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/blue/jshin2/zhangzihao/MML}"
export MML_ARTIFACT_ROOT="$PWD/artifacts" PYTHONUNBUFFERED=1
OUT=${OUT:-artifacts/motion/fair_v1}
W=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
STAGES=${STAGES:-"data train calibrate test summarize"}
PKG=mml_research.motion.fair_retrain
mkdir -p "$OUT/logs"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), stages: $STAGES, workers $W, commit $(git rev-parse HEAD)"
nvidia-smi -L || true

run() { echo -e "\n===== $1 ($(date '+%F %T')) ====="; shift; "$@" 2>&1 | grep -v "FutureWarning\|state = torch.load\|weights_only"; }

for s in $STAGES; do
  case $s in
    data)      run data      python -m $PKG.data --out-dir $OUT ;;
    train)     run train     python -m $PKG.train --out-dir $OUT ;;
    calibrate) run calibrate python -m $PKG.evaluate calibrate --out-dir $OUT --workers $W ;;
    test)      run test      python -m $PKG.evaluate test --out-dir $OUT --workers $W ;;
    summarize) run summarize python -m $PKG.summarize --out-dir $OUT ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
echo "all stages done $(date '+%F %T')"
