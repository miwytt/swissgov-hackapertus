"""
Evaluate all Apertus runs (+ GPT-4o reference) per split; write markdown and ACL-style LaTeX tables to results/.

python -m scripts.make_apertus_tables
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results"
PY = sys.executable
LANGS = ["de", "fr", "it"]
SPLITS = ["dev", "test"]
PRED = "data/evaluation/llm_predictions"

# (row label, json mode label, prefix per split)
ROWS = [("GPT-4o (paper)", "on", {s: f"{PRED}/full/gpt-4o-2024-08-06_admin_" for s in SPLITS})]
for size in ["8B", "70B"]:
    for suffix, mode in [("", "off"), ("-json", "on")]:
        ROWS.append((f"Apertus-v1.5-{size}", mode, {s: f"{PRED}/{s}/swiss-ai_Apertus-v1.5-{size}{suffix}_admin_" for s in SPLITS}))

from evaluation.utils import load_predictions  # noqa: E402

drop = set((ROOT / "list_to_drop.txt").read_text().split())


def spearman(prefix, split):
    out = subprocess.run([PY, "-m", "scripts.evaluate_predictions_admin", prefix, "--split", split],
                         cwd=ROOT, capture_output=True, text=True)
    lines = out.stdout.splitlines()
    try:
        i = lines.index("Spearman:")
    except ValueError:
        sys.exit(f"evaluation failed for {prefix} {split}:\n{out.stderr[-2000:]}")
    assert lines[i + 1].split("\t") == LANGS
    return [float(v.replace(",", ".")) for v in lines[i + 2].split("\t")]


def parse_rate(prefix):
    """share of items whose output the (strict) evaluator could parse into JSON"""
    preds = [p for lang in LANGS for p in load_predictions(ROOT / f"{prefix}{lang}.jsonl") if p.item_id not in drop]
    return 100 * sum(bool(p.get_json()) for p in preds) / len(preds)


def main():
    results = []
    for label, mode, prefixes in ROWS:
        scores = {s: spearman(prefixes[s], s) for s in SPLITS}
        rates = {s: parse_rate(prefixes[s]) for s in SPLITS}
        results.append((label, mode, scores, rates))
        print(label, mode, scores, rates)

    OUT.mkdir(exist_ok=True)

    # Markdown
    md = ["Spearman correlation (token-level, SwissGov-RSD, strict parsing as in the paper). "
          "Parsed = % of items with parseable JSON output (dev / test; GPT-4o: full set).", "",
          "| Model | JSON mode | Dev DE | Dev FR | Dev IT | Test DE | Test FR | Test IT | Parsed % (dev / test) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, mode, scores, rates in results:
        cells = [f"{v:.3f}" for s in SPLITS for v in scores[s]]
        md.append(f"| {label} | {mode} | " + " | ".join(cells) + f" | {rates['dev']:.1f} / {rates['test']:.1f} |")
    (OUT / "apertus_results_markdown.txt").write_text("\n".join(md) + "\n")

    # ACL-style LaTeX (booktabs)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{Dev} & \multicolumn{3}{c}{Test} \\",
           r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
           r"Model & JSON mode & DE & FR & IT & DE & FR & IT \\", r"\midrule"]
    for i, (label, mode, scores, _) in enumerate(results):
        cells = [f"{v:.3f}" for s in SPLITS for v in scores[s]]
        tex.append(f"{label} & {mode} & " + " & ".join(cells) + r" \\")
        if i == 0:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Spearman correlation between predicted and gold token-level difference labels on SwissGov-RSD "
            r"for prompt-based Apertus models (via the CSCS inference API), with and without constrained JSON decoding. "
            r"Unparseable outputs are scored as no difference, as in the original evaluation.}",
            r"\label{tab:apertus-results}", r"\end{table*}"]
    (OUT / "apertus_results_latex.txt").write_text("\n".join(tex) + "\n")
    print("wrote", OUT / "apertus_results_markdown.txt", OUT / "apertus_results_latex.txt")


if __name__ == "__main__":
    main()
