#!/bin/bash
#SBATCH --account=AIFAC_F02_1463
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --output=/leonardo_work/AIFAC_F02_1463/fcermell/vlmeval/logs/%x_%j.log
# env: MODE (mcq|caption). Serves Tarsier2-7B with the caption bake-off recipe (vLLM 0.23 sandbox,
# fcermell/bakeoff/scripts/{serve_tarsier.sh,lib.sh}), then runs haic_server_client.py; output in $R/haiccap/.
source /leonardo_work/AIFAC_F02_1463/fcermell/bakeoff/scripts/lib.sh
R=$W/fcermell/vlmeval
singularity exec --nv --bind /leonardo_work,/leonardo_scratch "${SENV[@]}" \
  --env MODEL=$MTAR --env PORT=8001 --env MEDIA_ROOT=$W/datasets/haic \
  $C23 bash $B/scripts/serve_tarsier.sh > $R/logs/serve_tarsier2_haic_${SLURM_JOB_ID}.log 2>&1 &
PID=$!
wait_ready tarsier2 8001 $PID 1500 || exit 1
OUTF=$R/haiccap/tarsier2_haic_${MODE}.jsonl
$PY $R/VLMEvalKit/scripts/sentune/haic/haic_server_client.py $MODE 8001 $OUTF
echo "[$(date +%T)] rows $(wc -l < $OUTF), errors $(grep -c '"error"' $OUTF)"
kill $PID
