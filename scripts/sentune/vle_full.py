"""Full-benchmark run, sharded one process per GPU, resumable.

usage: python vle_full.py run   MODEL OUTDIR SHARD NSHARDS DATASET [DATASET ...]
       python vle_full.py score MODEL OUTDIR NSHARDS DATASET [DATASET ...]

Rows are sharded by source video (crc32), so two processes never convert the same clip.
Each shard appends to OUTDIR/MODEL_DATASET.shardK.jsonl; `score` merges and runs evaluate().
"""
import json
import os
import sys
import time
import zlib

import pandas as pd

from vlmeval.dataset import HAIC, HAIC_Caption, MotionBench, MVBench

DATASETS = {
    'MVBench_64frame': lambda: MVBench(dataset='MVBench', nframe=64),
    'MotionBench': lambda: MotionBench(dataset='MotionBench'),
    'HAIC': lambda: HAIC(dataset='HAIC'),
    'HAIC_Caption': lambda: HAIC_Caption(dataset='HAIC_Caption'),
}


def shard_path(out, model, dname, k):
    return os.path.join(out, f'{model}_{dname}.shard{k}.jsonl')


def run(model_name, out, shard, nshards, dnames):
    from vlmeval.config import supported_VLM
    os.makedirs(out, exist_ok=True)
    model = supported_VLM[model_name]()
    for dname in dnames:
        ds = DATASETS[dname]()
        df = ds.data
        vid = df['prefix'].astype(str) + '/' + df['video'] if 'prefix' in df else df['video']
        df = df[vid.map(lambda v: zlib.crc32(v.encode()) % nshards) == shard]
        f = shard_path(out, model_name, dname, shard)
        done = set()
        if os.path.exists(f):
            done = {json.loads(line)['index'] for line in open(f)}
        todo = df[~df['index'].isin(done)]
        print(f'{dname} shard {shard}/{nshards}: {len(df)} rows, {len(todo)} to do', flush=True)
        t0 = time.time()
        with open(f, 'a') as fh:
            for i, (_, row) in enumerate(todo.iterrows()):
                t = time.time()
                try:
                    pred = model.generate(ds.build_prompt(row, video_llm=model.VIDEO_LLM), dataset=ds.dataset_name)
                except Exception as e:  # one bad clip must not kill the shard
                    print(f'  index {row["index"]}: {type(e).__name__}: {e}', flush=True)
                    continue
                fh.write(json.dumps({'index': int(row['index']), 'prediction': pred,
                                     'seconds': round(time.time() - t, 2)}) + '\n')
                fh.flush()
                if (i + 1) % 100 == 0:
                    print(f'  {i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/q', flush=True)


def score(model_name, out, nshards, dnames):
    for dname in dnames:
        ds = DATASETS[dname]()
        preds = pd.concat([pd.read_json(shard_path(out, model_name, dname, k), lines=True)
                           for k in range(nshards)]).drop_duplicates('index', keep='last')
        df = ds.data.merge(preds, on='index', how='left').sort_values('index').reset_index(drop=True)
        assert (df['index'] == df.index).all(), 'evaluate() needs index == row label'
        print(f'{dname}: {df["prediction"].notna().sum()}/{len(df)} predicted, '
              f'{df["seconds"].mean():.2f}s/q', flush=True)
        f = os.path.join(out, f'{model_name}_{dname}.tsv')
        df.to_csv(f, sep='\t', index=False)
        print(f'{dname} rating: {ds.evaluate(f, model="exact_matching")}', flush=True)


if __name__ == '__main__':
    mode, model_name, out = sys.argv[1:4]
    if mode == 'run':
        run(model_name, out, int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:])
    else:
        score(model_name, out, int(sys.argv[4]), sys.argv[5:])
