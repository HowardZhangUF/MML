#!/bin/bash
# WOMD benchmark on HiPerGator: data -> train + forecast (GPU) | calibrate -> test -> summarize (CPU burst).
#   S=mml_research/src/mml_research/motion/womd/womd.sbatch
#   A=$(sbatch --parsable --qos=jshin2 --partition=hpg-turin --gpus=1 --cpus-per-task=8 --mem=64gb \
#        --time=2-00:00:00 --export=ALL,STAGES="data train" $S)
#   sbatch --dependency=afterok:$A --qos=jshin2-b --cpus-per-task=96 --mem=320gb --time=2-00:00:00 \
#        --export=ALL,STAGES="calibrate test summarize" $S
#SBATCH --job-name=womd
#SBATCH --account=jshin2
#SBATCH --qos=jshin2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/motion/womd_v1/logs/slurm_womd_%j.log

# /etc/profile and conda's hook are not `set -u` clean; enable strict mode after activation.
source /etc/profile
module load conda
eval "$(conda shell.bash hook)"
conda activate zhz14
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/blue/jshin2/zhangzihao/MML}"
export MML_ARTIFACT_ROOT="$PWD/artifacts" PYTHONUNBUFFERED=1
OUT=${OUT:-artifacts/motion/womd_v1}
W=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
STAGES=${STAGES:-"data train calibrate test summarize"}
PKG=mml_research.motion.womd
mkdir -p "$OUT/logs"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), stages: $STAGES, workers $W, commit $(git rev-parse HEAD)"
nvidia-smi -L || true

run() { echo -e "\n===== $1 ($(date '+%F %T')) ====="; shift; "$@" 2>&1 | grep --line-buffered -v "FutureWarning\|state = torch.load\|weights_only"; }

for s in $STAGES; do
  case $s in
    data)      run data      python -m $PKG.data --out-dir $OUT --workers $W ;;
    train)     run train     python -m $PKG.train --out-dir $OUT ;;
    calibrate) run calibrate python -m $PKG.evaluate calibrate --out-dir $OUT --workers $W ;;
    test)      run test      python -m $PKG.evaluate test --out-dir $OUT --workers $W ;;
    summarize) run summarize python -m $PKG.summarize --out-dir $OUT ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
echo "all stages done $(date '+%F %T')"
