"""MotionBench (CVPR 2025): fine-grained motion understanding, multiple choice.

Paper: https://arxiv.org/abs/2501.02955 · Data: https://huggingface.co/datasets/zai-org/MotionBench

Only the dev split is scored: test answers are "NA" (leaderboard only). Questions whose clip is
not in the HF repo are dropped and counted in a warning (in the 2025 release these are the
zero-length public-dataset spans, start == end).

Set MOTIONBENCH_ROOT to a local copy of the HF repo (the folder holding MotionBench/) to run
offline; otherwise the repo is fetched with snapshot_download.
"""
import json
import os
import os.path as osp
import re
import warnings

import pandas as pd
from huggingface_hub import snapshot_download

from ..smp import dump, load
from ..smp.file import get_file_extension, get_intermediate_file_path
from ..utils import can_infer
from .video_base import VideoBaseDataset

OPTION_RE = re.compile(r'^([A-F])\.\s*(.*)$')


class MotionBench(VideoBaseDataset):

    TYPE = 'Video-MCQ'
    POST_PROMPT = "Answer with the option's letter from the given choices directly."

    def __init__(self, dataset='MotionBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MotionBench']

    def prepare_dataset(self, dataset_name='MotionBench', repo_id='zai-org/MotionBench'):
        root = os.environ.get('MOTIONBENCH_ROOT') or snapshot_download(repo_id=repo_id, repo_type='dataset')
        bench = osp.join(root, 'MotionBench')
        data_file = osp.join(bench, f'{dataset_name}.tsv')
        if not osp.exists(data_file):
            self._build_tsv(bench, data_file)
        return dict(root=bench, data_file=data_file)

    @staticmethod
    def _build_tsv(bench, data_file):
        rows, missing = [], 0
        with open(osp.join(bench, 'video_info.meta.jsonl')) as f:
            for line in f:
                meta = json.loads(line)
                for qa in meta['qa']:
                    if qa['answer'] == 'NA':
                        continue
                    lines = [x.strip() for x in qa['question'].split('\n') if x.strip()]
                    options = [OPTION_RE.match(x).groups() for x in lines if OPTION_RE.match(x)]
                    stem = ' '.join(x for x in lines if not OPTION_RE.match(x))
                    sub = 'public-dataset' if qa['start'] is not None else 'self-collected'
                    video = meta['video_path'] if sub == 'self-collected' \
                        else f"{meta['key']}_{qa['start']}_{qa['end']}.mp4"
                    if not osp.exists(osp.join(bench, sub, video)):
                        missing += 1
                        continue
                    rows.append(dict(
                        uid=qa['uid'], question_type=meta['question_type'], video_type=meta['video_type'],
                        video=f'{sub}/{video[:-4]}', question=stem,
                        candidates=json.dumps(dict(options)), answer=qa['answer'],
                    ))
        if missing:
            warnings.warn(f'MotionBench: dropped {missing} dev questions whose clip is not in the repo')
        df = pd.DataFrame(rows)
        df['index'] = range(len(df))
        df.to_csv(data_file, sep='\t', index=False)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        options = json.loads(line['candidates'])
        question = line['question'] + '\n' + '\n'.join(f'{k}. {v}' for k, v in options.items())
        message = []
        if video_llm or not self.split_frame:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video'] + '.mp4')))
        else:
            message += [dict(type='image', value=p) for p in self.save_video_frames(line['video'])]
        message.append(dict(type='text', value=f'{question}\n{self.POST_PROMPT}'))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv']
        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        data = load(eval_file)
        data['pred'] = [
            can_infer(str(p), json.loads(c)) or 'Z' for p, c in zip(data['prediction'], data['candidates'])
        ]
        data['score'] = (data['pred'] == data['answer']).astype(int)
        dump(data, score_file)
        rating = {qt: round(g['score'].mean() * 100, 2) for qt, g in data.groupby('question_type')}
        rating['overall'] = round(data['score'].mean() * 100, 2)
        rating['n'] = int(len(data))
        rating['unparsed'] = int((data['pred'] == 'Z').sum())
        dump(rating, rating_file)
        return rating
