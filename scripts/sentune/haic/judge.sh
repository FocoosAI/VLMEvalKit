#!/bin/bash
#SBATCH --account=AIFAC_F02_1463
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1 --cpus-per-task=16 --gres=gpu:2 --mem=120G -t 00:30:00
# HAIC caption setting, step 2: text-only judge answers the 2,000 test questions from captions.
# env: CAPS (captions jsonl), OUT (judgement jsonl), optional CAPTSV (HAIC_Caption prediction tsv -> CAPS first). Paths relative to $R.
R=/leonardo_work/AIFAC_F02_1463/fcermell/vlmeval; C=/leonardo_work/AIFAC_F02_631/containers/vllm_sandbox_v0260
cd $R
[ -n "${CAPTSV:-}" ] && singularity exec --bind /leonardo_work $C venv/bin/python VLMEvalKit/scripts/sentune/haic/tsv2caps.py $CAPTSV $CAPS
singularity exec --nv --bind /leonardo_work --env HF_HOME=/leonardo_work/AIFAC_F02_1463/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 \
  --env VLLM_CACHE_ROOT=/leonardo_work/AIFAC_F02_1463/.cache/vllm --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  $C bash $R/_inner.sh venv/bin/python VLMEvalKit/scripts/sentune/haic/haic_caption_judge.py \
  /leonardo_work/AIFAC_F02_1463/datasets/haic/haic_bench_v2_test.json $CAPS $OUT --model /leonardo_work/AIFAC_F02_631/.cache/huggingface/hub/models--QuantTrio--Qwen3.6-35B-A3B-AWQ/snapshots/119886a1072372348f73ef0df2d801cdcc0f455b
