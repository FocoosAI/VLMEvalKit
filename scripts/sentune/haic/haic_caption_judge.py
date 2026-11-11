"""HAICBench caption setting: a text-only LLM answers the 2,000 test MCQ from one caption per video.

Prompt = the paper's Figure 11 (arXiv 2502.20811v2) plus a last line asking for the letter only
(without it the local judge reasons first and the answer letter is ambiguous). The paper's answerer is Gemini-1.5-Pro;
here it is a local LLM (default QuantTrio/Qwen3.6-35B-A3B-AWQ, thinking off, greedy).
A refusal ("The caption does not mention it.") or an unparsable answer counts as wrong.

    python haic_caption_judge.py TEST_JSON CAPTIONS_JSONL OUT_JSONL [--model M] [--tp 2]

CAPTIONS_JSONL rows: {"video": "<file>.mp4", "caption": "..."}; video matches the test JSON.
"""
import argparse, json, re
from collections import defaultdict

PROMPT = ('{caption}\nAccording to the video caption above, answer the following question. If the given caption '
          'doesn\'t provide sufficient information to answer the question, just reply, "The caption does not '
          'mention it." Question:\n{question}\nOptions:\n{options}\n'
          'Reply with only the letter of the correct option, or with "The caption does not mention it."')
LETTER = re.compile(r'^\W*\(?([A-D])\b')
ANY = re.compile(r'\b([A-D])\.')

def parse(t):
    t = re.sub(r'<think>.*?</think>', '', t, flags=re.S).strip()
    if 'does not mention' in t.lower():
        return 'refusal'
    m = LETTER.match(t) or (ANY.findall(t) or [None])[-1]
    if isinstance(m, str):
        return m
    return m.group(1) if m else 'unparsed'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('test'); ap.add_argument('captions'); ap.add_argument('out')
    ap.add_argument('--model', default='QuantTrio/Qwen3.6-35B-A3B-AWQ'); ap.add_argument('--tp', type=int, default=2)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    caps = {r['video']: r['caption'] for r in map(json.loads, open(a.captions))}
    qs = [q for q in json.load(open(a.test))]
    msgs = []
    for q in qs:
        opts = '\n'.join(f'{chr(65 + i)}. {c}' for i, c in enumerate(q['candidates']))
        cap = caps.get(q['video']) or ''
        msgs.append([{'role': 'user', 'content': PROMPT.format(caption=cap, question=q['original_question'], options=opts)}])
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=8192, gpu_memory_utilization=0.9)
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=512),
                    chat_template_kwargs={'enable_thinking': False})
    by = defaultdict(list); n_ok = 0
    with open(a.out, 'w') as f:
        for q, o in zip(qs, outs):
            t = o.outputs[0].text; p = parse(t); ok = p == q['answer']
            n_ok += ok; by[q['subfield']].append(ok)
            f.write(json.dumps({'video': q['video'], 'category': q['subfield'], 'question': q['original_question'],
                                'answer': q['answer'], 'judge_output': t, 'pred': p, 'correct': ok,
                                'has_caption': bool(caps.get(q['video']))}, ensure_ascii=False) + '\n')
    res = {c: round(100 * sum(v) / len(v), 2) for c, v in sorted(by.items())}
    res['avg_categories'] = round(sum(res.values()) / len(res), 2)
    res['overall'] = round(100 * n_ok / len(qs), 2)
    preds = [json.loads(l)['pred'] for l in open(a.out)]
    res['refusal'] = round(100 * preds.count('refusal') / len(preds), 2)
    res['unparsed'] = preds.count('unparsed')
    print(json.dumps(res)); json.dump(res, open(a.out.replace('.jsonl', '.summary.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
