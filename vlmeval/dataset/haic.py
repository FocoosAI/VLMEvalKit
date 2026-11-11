"""HAICBench v2 (ACL 2025): human action understanding, multiple choice, "standard" setting.

Paper: https://arxiv.org/abs/2502.20811 · Data: https://huggingface.co/datasets/KuaishouHAIC/HAIC

Scores the test split: 2,000 four-option questions on 412 videos, 400 per category (Interaction,
Attribute, Count, Sequential, Details). The prompt is the question, the lettered options and the
kit's letter instruction, as for MotionBench.

Set HAIC_ROOT to a local copy of the HF repo with videos_v2.zip extracted (offline); otherwise the
repo is fetched with snapshot_download, and the zip must still be extracted by hand.
"""
import json
import os
import os.path as osp
import warnings

import pandas as pd
from huggingface_hub import snapshot_download

from ..smp import dump, load
from ..smp.file import get_file_extension, get_intermediate_file_path
from ..utils import can_infer
from .video_base import VideoBaseDataset


class HAIC(VideoBaseDataset):

    TYPE = 'Video-MCQ'
    POST_PROMPT = "Answer with the option's letter from the given choices directly."

    def __init__(self, dataset='HAIC', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['HAIC']

    def prepare_dataset(self, dataset_name='HAIC', repo_id='KuaishouHAIC/HAIC'):
        root = os.environ.get('HAIC_ROOT') or snapshot_download(repo_id=repo_id, repo_type='dataset')
        data_file = osp.join(root, f'{dataset_name}.tsv')
        if not osp.exists(data_file):
            self._build_tsv(root, data_file)
        return dict(root=root, data_file=data_file)

    @staticmethod
    def _build_tsv(root, data_file):
        videos = {}
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith('.mp4'):
                    videos.setdefault(f, osp.relpath(osp.join(d, f), root))
        rows, missing = [], 0
        for qa in json.load(open(osp.join(root, 'haic_bench_v2_test.json'))):
            if qa['video'] not in videos:
                missing += 1
                continue
            rows.append(dict(
                question_type=qa['subfield'], video=videos[qa['video']][:-4],
                question=qa['original_question'],
                candidates=json.dumps({chr(65 + i): c for i, c in enumerate(qa['candidates'])}),
                answer=qa['answer'],
            ))
        if missing:
            warnings.warn(f'HAIC: dropped {missing} test questions whose clip is not on disk')
        df = pd.DataFrame(rows)
        df['index'] = range(len(df))
        tmp = f'{data_file}.{os.getpid()}.tmp'   # shards build concurrently: write, then rename
        df.to_csv(tmp, sep='\t', index=False)
        os.replace(tmp, data_file)

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


class HAIC_Caption(HAIC):
    """HAICBench caption setting, step 1: one people-centric caption per test video (412 rows).

    The prompt is the paper's "Prompt for MLLM Caption" (Figure 11). Step 2 (a text-only LLM
    answering the 2,000 test questions from each caption) is scripts/sentune/haic/haic_caption_judge.py;
    evaluate() here only reports caption counts and lengths.
    """

    TYPE = 'Video-VQA'
    PROMPT = ('Please provide a description of people in the video, your description should describe their '
              'attributes and detailed actions in chronological order. Do not mention anything unrelated to people.')

    def __init__(self, dataset='HAIC_Caption', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['HAIC_Caption']

    @staticmethod
    def _build_tsv(root, data_file):
        mcq = f'{data_file}.{os.getpid()}.mcq'
        HAIC._build_tsv(root, mcq)
        df = pd.read_csv(mcq, sep='\t')
        os.remove(mcq)
        df = df.drop_duplicates('video')[['video']].reset_index(drop=True)
        df['question'] = HAIC_Caption.PROMPT
        df['index'] = range(len(df))
        tmp = f'{data_file}.{os.getpid()}.tmp'
        df.to_csv(tmp, sep='\t', index=False)
        os.replace(tmp, data_file)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        message = []
        if video_llm or not self.split_frame:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video'] + '.mp4')))
        else:
            message += [dict(type='image', value=p) for p in self.save_video_frames(line['video'])]
        message.append(dict(type='text', value=self.PROMPT))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        words = data['prediction'].astype(str).str.split().str.len()
        rating = {'n': int(len(data)), 'empty': int((words == 0).sum()),
                  'words_median': float(words.median()), 'words_max': int(words.max())}
        dump(rating, get_intermediate_file_path(eval_file, '_rating', 'json'))
        return rating
