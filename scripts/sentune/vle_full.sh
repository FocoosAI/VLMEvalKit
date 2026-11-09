#!/bin/bash
#SBATCH --account=AIFAC_F02_1463
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=0
#SBATCH --time=03:00:00
# Full VLMEvalKit run: 4 shards (one per GPU) then merge+score. Resumable: resubmit to continue.
# env: MODEL (supported_VLM key), DATASETS (space-separated), OUT (dir under $R),
#      GPUS_PER_SHARD (default 1; the model uses tensor parallel over its shard's GPUs),
#      VENV (default venv; venv-lfm3 for models needing a newer transformers)
# Expects $R/VLMEvalKit (this repo) and $R/venv (container python + VLMEvalKit, --system-site-packages).
set -u
R=/leonardo_work/AIFAC_F02_1463/fcermell/vlmeval
C=/leonardo_work/AIFAC_F02_631/containers/vllm_sandbox_v0260
D=/leonardo_work/AIFAC_F02_1463/datasets
G=${GPUS_PER_SHARD:-1}
N=$((4 / G))
ENVS=(--env VLLM_NO_USAGE_STATS=1 --env HF_HOME=/leonardo_work/AIFAC_F02_1463/.cache/huggingface --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env VLLM_CACHE_ROOT=/leonardo_work/AIFAC_F02_1463/.cache/vllm
      --env VLLM_WORKER_MULTIPROC_METHOD=spawn --env LMUData=$R/LMUData
      --env MVBENCH_ROOT=$D/mvbench --env MOTIONBENCH_ROOT=$D/motionbench --env HAIC_ROOT=$D/haic)
# The container's CUDA forward-compat libs must come first (Leonardo driver is CUDA 12.2).
cat > $R/_inner.sh <<'INNER'
COMPAT=$(ls -d /usr/local/cuda-*/compat 2>/dev/null | sort -V | tail -1)
export LD_LIBRARY_PATH=${COMPAT}:${LD_LIBRARY_PATH:-}
exec "$@"
INNER
mkdir -p $R/logs $R/LMUData
cd $R
PY=VLMEvalKit/scripts/sentune/vle_full.py
run() { local g=$1; shift; singularity exec --nv --bind /leonardo_work,/leonardo_scratch "${ENVS[@]}" \
          --env CUDA_VISIBLE_DEVICES=$g $C bash $R/_inner.sh "$@"; }
for k in $(seq 0 $((N-1))); do
  run $(seq -s, $((k*G)) $((k*G+G-1))) ${VENV:-venv}/bin/python $PY run $MODEL $OUT $k $N $DATASETS > logs/full_${SLURM_JOB_ID}_shard$k.log 2>&1 &
done
wait
run 0 ${VENV:-venv}/bin/python $PY score $MODEL $OUT $N $DATASETS
