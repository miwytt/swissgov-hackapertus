"""
Prompt-based SwissGov-RSD predictions via the CSCS (OpenAI-compatible) API, e.g. for Apertus.

Reuses the exact prompts of the paper's GPT-4o runs (they align 1:1 with the current gold files
after dropping LIST_TO_DROP), so results are directly comparable.

dev and test are separate runs with separate output files:
python -m scripts.predict_cscs --model swiss-ai/Apertus-v1.5-8B --base_url https://api.inference.cscs.ch/v1 --split dev --lang de
python -m scripts.evaluate_predictions_admin data/evaluation/llm_predictions/dev/swiss-ai_Apertus-v1.5-8B_admin_ --split dev
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
PRED_DIR = ROOT / "data" / "evaluation" / "llm_predictions" / "full"

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)  # swiss-ai/Apertus-v1.5-8B; swiss-ai/Apertus-v1.5-70B
parser.add_argument("--base_url", required=True)
parser.add_argument("--split", required=True, choices=["dev", "test"])
parser.add_argument("--lang", required=True, choices=["de", "fr", "it"])
parser.add_argument("--key_file", default=str(ROOT / ".keys" / "cscs.txt"))
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--max_tokens", type=int, default=16384, help="caps runaway/looping generations")
parser.add_argument("--json_mode", action="store_true", help="constrained JSON decoding; garbles non-ASCII tokens with Apertus")
parser.add_argument("--redo_errors", action="store_true", help="re-query items stored with an API error, in place")
parser.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
args = parser.parse_args()

client = OpenAI(api_key=open(args.key_file).read().strip(), base_url=args.base_url, timeout=900, max_retries=6)
drop = set(open(ROOT / "list_to_drop.txt").read().split())

source = PRED_DIR / f"gpt-4o-2024-08-06_admin_{args.lang}.jsonl"
items = [json.loads(line) for line in open(source)]
items = [x for x in items if x["item_id"] not in drop]
# kept prompts align positionally with the full gold file; select this split by gold id
gold_dir = ROOT / "data" / "evaluation" / "gold_labels"
full_ids = [json.loads(line)["id"] for line in open(gold_dir / "full" / f"gold_admin_{args.lang}.jsonl")]
split_ids = {json.loads(line)["id"] for line in open(gold_dir / args.split / f"gold_admin_{args.lang}.jsonl")}
assert len(items) == len(full_ids)
items = [x for x, gid in zip(items, full_ids) if gid in split_ids][:args.limit]
assert args.limit or len(items) == len(split_ids)

output = PRED_DIR.parent / args.split / f"{args.model.replace('/', '_')}{'-json' if args.json_mode else ''}_admin_{args.lang}.jsonl"
output.parent.mkdir(exist_ok=True)
done = sum(1 for _ in open(output)) if output.exists() else 0  # file is always an in-order prefix -> resume


def predict(item):
    messages = item["api_request"]["body"]["messages"]
    params = {"temperature": 0, "max_tokens": args.max_tokens}
    if args.json_mode:
        params["response_format"] = {"type": "json_object"}
    for attempt in range(3):  # client retries don't cover a stream dropped mid-body
        content, finish_reason, usage, response_id, model, error = "", None, None, None, None, None
        try:
            # streamed so long (looping) generations don't hit the gateway's 504 upstream timeout
            stream = client.chat.completions.create(model=args.model, messages=messages, stream=True,
                                                    stream_options={"include_usage": True}, **params)
            for chunk in stream:
                response_id, model = chunk.id, chunk.model
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices:
                    content += chunk.choices[0].delta.content or ""
                    finish_reason = chunk.choices[0].finish_reason or finish_reason
            break
        except Exception as e:  # after all retries: keep the run going, item is scored like unparseable output
            error = f"{type(e).__name__}: {e}"
            tqdm.write(f"{item['item_id']} (attempt {attempt + 1}): {error}")
    return {
        "item_id": item["item_id"],
        "sentence1": item["sentence1"],
        "sentence2": item["sentence2"],
        "api_request": {
            "custom_id": f"request-{item['item_id']}",
            "body": {"model": args.model, "messages": messages, **params},
        },
        "api_response": {
            "id": response_id,
            "response": {
                "status_code": 200 if error is None else 500,
                "body": {
                    "id": response_id,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }],
                    "usage": usage,
                },
            },
            "error": error,
        },
        "provider": "openai",  # same response layout as OpenAI -> parsed by evaluation.predictions.LLMPrediction
    }


if args.redo_errors:
    preds = [json.loads(line) for line in open(output)]
    redo = [i for i, p in enumerate(preds) if p["api_response"]["error"] is not None]
    with ThreadPoolExecutor(args.workers) as pool:
        for i, pred in zip(redo, tqdm(pool.map(predict, [items[i] for i in redo]), total=len(redo), desc=f"redo {args.lang}")):
            assert pred["item_id"] == preds[i]["item_id"]
            preds[i] = pred
    output.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
    print(f"redid {len(redo)} items in {output}, still failing: {sum(p['api_response']['error'] is not None for p in preds)}")
    sys.exit()

with open(output, "a") as f_out, ThreadPoolExecutor(args.workers) as pool:
    # pool.map yields in input order, so the file stays positionally aligned with the gold labels
    for pred in tqdm(pool.map(predict, items[done:]), total=len(items) - done, desc=f"{args.model} {args.lang}"):
        f_out.write(json.dumps(pred, ensure_ascii=False) + "\n")
        f_out.flush()

print(f"Results saved to {output}")
