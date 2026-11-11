"""HAICBench via an OpenAI-compatible vLLM server (for models the VLMEvalKit runner can't load, e.g. Tarsier2).
Same prompts as vlmeval/dataset/haic.py. Resumable.
  mcq:     python haic_server_client.py mcq PORT OUT.jsonl     -> {index, category, answer, prediction}
  caption: python haic_server_client.py caption PORT OUT.jsonl -> {video, caption} (judge input format)"""
import asyncio, json, os, re, sys, time
import pandas as pd
from openai import AsyncOpenAI
mode, port, out = sys.argv[1], sys.argv[2], sys.argv[3]
D = '/leonardo_work/AIFAC_F02_1463/datasets/haic'
POST = "Answer with the option's letter from the given choices directly."
CAP = ('Please provide a description of people in the video, your description should describe their attributes and '
       'detailed actions in chronological order. Do not mention anything unrelated to people.')
df = pd.read_csv(f'{D}/HAIC.tsv', sep='\t').sort_values(['video', 'index'])  # clip-major for prefix caching
if mode == 'caption':
    df = df.drop_duplicates('video')
key = 'index' if mode == 'mcq' else 'video'
done = set()
if os.path.exists(out):
    done = {json.loads(l)[key] for l in open(out) if 'error' not in json.loads(l)}
todo = [r for _, r in df.iterrows() if (int(r['index']) if mode == 'mcq' else os.path.basename(r['video']) + '.mp4') not in done]
cli = AsyncOpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key='x', timeout=900)
sem = asyncio.Semaphore(16); fh = open(out, 'a')
async def one(r):
    if mode == 'mcq':
        opts = json.loads(r['candidates'])
        text = r['question'] + '\n' + '\n'.join(f'{k}. {v}' for k, v in opts.items()) + '\n' + POST
        rec = {'index': int(r['index']), 'category': r['question_type'], 'answer': r['answer']}; mt = 64
    else:
        text = CAP; rec = {'video': os.path.basename(r['video']) + '.mp4'}; mt = 1024
    async with sem:
        try:
            resp = await cli.chat.completions.create(model='model', max_tokens=mt, temperature=0, messages=[{'role': 'user', 'content': [
                {'type': 'video_url', 'video_url': {'url': f"file://{D}/{r['video']}.mp4"}}, {'type': 'text', 'text': text}]}])
            c = (resp.choices[0].message.content or '').strip()
            rec['prediction' if mode == 'mcq' else 'caption'] = c
            rec['prompt_tokens'] = resp.usage.prompt_tokens
        except Exception as e:
            rec['error'] = repr(e)[:300]
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
async def main():
    t0 = time.time(); await asyncio.gather(*[one(r) for r in todo])
    print(f'WALL {mode} n={len(todo)} {time.time()-t0:.0f}s', flush=True)
asyncio.run(main())
