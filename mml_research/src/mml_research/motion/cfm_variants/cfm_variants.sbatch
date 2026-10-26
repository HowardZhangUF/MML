#!/bin/bash
# Full CFM-variant pipeline on HiPerGator (guide 11.2 order):
#   audit -> prepare -> train -> verify -> calibrate(+lock) -> test -> dev -> summarize
# Submit from the repo root as two dependent jobs (the GPU is only needed for training):
#   S=mml_research/src/mml_research/motion/cfm_variants/cfm_variants.sbatch
#   A=$(sbatch --parsable --qos=jshin2 --partition=hpg-rtx6000 --gpus=1 --cpus-per-task=16 \
#        --export=ALL,STAGES="audit prepare train verify" $S)
#   sbatch --dependency=afterok:$A --qos=jshin2-b --cpus-per-task=32 \
#        --export=ALL,STAGES="calibrate test dev summarize" $S
# Env overrides: STAGES="train verify" OUT=... WORKERS=16
#SBATCH --job-name=cfm_variants
#SBATCH --account=jshin2
#SBATCH --qos=jshin2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/motion/cfm_variants_v1/logs/slurm_cfm_variants_%j.log

# /etc/profile and conda's hook are not `set -u` clean; enable strict mode after activation.
source /etc/profile
module load conda
eval "$(conda shell.bash hook)"
conda activate zhz14
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/blue/jshin2/zhangzihao/MML}"
export MML_ARTIFACT_ROOT="$PWD/artifacts" PYTHONUNBUFFERED=1
M=artifacts/motion/exit_benchmark/models
OUT=${OUT:-artifacts/motion/cfm_variants_v1}
W=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
STAGES=${STAGES:-"audit prepare train verify calibrate test dev summarize"}
PKG=mml_research.motion.cfm_variants
mkdir -p "$OUT/logs"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), stages: $STAGES, workers $W, commit $(git rev-parse HEAD)"
nvidia-smi -L || true

run() { echo -e "\n===== $1 ($(date '+%F %T')) ====="; shift; "$@" 2>&1 | grep -v "FutureWarning\|state = torch.load"; }

for s in $STAGES; do
  case $s in
    audit)     run audit     python -m $PKG.audit --models-dir $M --out-dir $OUT ;;
    prepare)   run prepare   python -m $PKG.prepare --models-dir $M --out-dir $OUT --workers $W ;;
    train)     run train     python -m $PKG.train --models-dir $M --out-dir $OUT --methods all ;;
    verify)    run verify    python -m $PKG.verify --models-dir $M --out-dir $OUT ;;
    calibrate) run calibrate python -m $PKG.calibrate --models-dir $M --out-dir $OUT --workers $W ;;
    test)      run test      python -m $PKG.run --split test --models-dir $M --out-dir $OUT --workers $W ;;
    dev)       run dev       python -m $PKG.run --split dev --methods PF+CFM PF+DMMN PF+NF \
                               --models-dir $M --out-dir $OUT --workers $W ;;
    summarize) run summarize python -m $PKG.summarize --split test --out-dir $OUT
               run summarize-dev python -m $PKG.summarize --split dev --out-dir $OUT \
                               --old-results artifacts/motion/exit_benchmark/hpg/results_all13 ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
echo "all stages done $(date '+%F %T')"
