"""HAIC_Caption prediction TSV -> captions jsonl for haic_caption_judge.py."""
import json, os, sys, pandas as pd
df = pd.read_csv(sys.argv[1], sep='\t')
with open(sys.argv[2], 'w') as f:
    for v, p in zip(df['video'], df['prediction']):
        f.write(json.dumps({'video': os.path.basename(str(v)) + '.mp4', 'caption': '' if pd.isna(p) else str(p)}, ensure_ascii=False) + '\n')
print(len(df), 'captions ->', sys.argv[2])
