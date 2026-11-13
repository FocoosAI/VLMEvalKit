"""Write <stem>_rescored_rating.json next to the kit's <stem>_rating.json (MVBench, MotionBench).

Same keys and shape as the kit file. Scores use the final-answer override: the letter after the
LAST 'best/correct/final option/answer/choice' replaces the kit's exact_matching score; rows
without that phrase keep the kit score. Usage: rescore_rating.py <stem>.tsv [...]
"""
import json
import re
import sys

import pandas as pd

PAT = re.compile(r"(?i:best|correct|final)\s*(?i:option|answer|choice)\s*(?i:is)?\s*[*:\s]*\(?\s*([A-F])\s*(?:\)|\.|\*|,|$|\s)")


def final(p):
    m = PAT.findall(str(p))
    return m[-1] if m else None


def rescore(tsv):
    stem = tsv[:-len('.tsv')]
    df = pd.read_csv(tsv, sep='\t')
    sc = pd.read_excel(stem + '_score.xlsx').set_index('index').loc[df['index']]
    with open(stem + '_rating.json') as f:
        kit = json.load(f)
    df['kit'] = sc['score'].values
    if 'task_type' in df:  # MVBench: answer is the option text
        df['gold'] = df.apply(lambda r: chr(65 + eval(r['candidates']).index(r['answer'])), axis=1)
        key = 'task_type'
    else:  # MotionBench: answer is the letter
        df['gold'] = df['answer']
        key = 'question_type'
    fin = df.prediction.map(final)
    df['new'] = df.kit.where(fin.isna(), (fin == df.gold).astype(int))

    def build(col):
        out = {}
        for k, v in kit.items():
            g = df if k == 'overall' else df[df[key] == k]
            if isinstance(v, list):
                c, n = int(g[col].sum()), len(g)
                out[k] = [c, n, f'{c / n * 100:.2f}%']
            elif k == 'n':
                out[k] = len(df)
            elif k == 'unparsed':
                z = sc['pred'].values == 'Z'
                out[k] = int(z.sum()) if col == 'kit' else int((z & fin.isna().values).sum())
            else:
                out[k] = round(g[col].mean() * 100, 2)
        return out

    assert build('kit') == kit, f'{tsv}: cannot reproduce the kit rating from its score file'
    out = build('new')
    with open(stem + '_rescored_rating.json', 'w') as f:
        json.dump(out, f, indent=4)
    return out, int((df.new != df.kit).sum())


if __name__ == '__main__':
    for tsv in sys.argv[1:]:
        out, changed = rescore(tsv)
        o = out['overall']
        print(f"{tsv.split('/')[-1]} overall={o} changed={changed}", flush=True)
