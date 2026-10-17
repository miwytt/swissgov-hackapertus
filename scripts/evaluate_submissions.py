"""
Import, validate and score the Hack Apertus Track 2A submissions (clones in submissions/, commits pinned in
submissions.tsv). Only each team's committed, self-designated predictions are read; no submission code is run.

1. import: copy each team's files into data/evaluation/submission_predictions/<team>/{full,dev,test}/ in the
   evaluator's EncoderPrediction format, after checking ids, tokens and label lengths against the gold files
2. score: official evaluator (scripts.evaluate_predictions_admin), dev and test separately
3. write results/submissions_results_{markdown,latex}.txt

python -m scripts.evaluate_submissions
"""
import json
import zipfile
from pathlib import Path

from scripts.make_apertus_tables import LANGS, OUT, ROOT, spearman

SUB = ROOT / "submissions"
DEST = ROOT / "data" / "evaluation" / "submission_predictions"
GOLD = ROOT / "data" / "evaluation" / "gold_labels"

# team -> {run: {split: source}}; the first run is the team's official submission (per its README).
# split "full" = 224 rows/lang in full-set order (the evaluator filters dev/test from it).
# source: "{lang}"-templated path under submissions/, or (zip path, "{lang}"-templated member).
# Excluded: byte-identical copies, gold/data files, RoestiBridge raw_* (predictions_* derive from them) and its
# 40-doc dev subsets, tsemin's data/reference (organisers' baselines).
TEAMS = {
    "apertus-swissgov-rsd": {
        "apertus_head": {"test": "apertus-swissgov-rsd/track_2a/data/predictions_test/apertus_head_{lang}.jsonl.jsonl"},
    },
    "clauseshift": {
        "CLAUSESHIFT": {"full": "clauseshift/track_2a/predictions/CLAUSESHIFT_admin_{lang}.jsonl.jsonl"},
    },
    "Hack-Apertus-Swiss-open-source-AI": {
        "diffalign": {
            "dev": ("Hack-Apertus-Swiss-open-source-AI/track_2a/predictions/swissgov_rsd_track2a_apertus_submission.zip", "diffalign_{lang}.jsonl"),
            "test": ("Hack-Apertus-Swiss-open-source-AI/track_2a/predictions/swissgov_rsd_track2a_apertus_submission_test.zip", "diffalign_{lang}.jsonl"),
        },
    },
    "hack-apertus-tsemin": {
        "tsemin (fuse verified+d4)": {"full": "hack-apertus-tsemin/track_2a/data/predictions/full/tsemin_admin_{lang}.jsonl.jsonl"},
        "b0s baseline": {"test": "hack-apertus-tsemin/track_2a/data/baselines/b0s/test/b0s_admin_{lang}.jsonl.jsonl"},
    },
    "polyjudge-swissgov-rsd": {
        "M1 (main)": {"test": "polyjudge-swissgov-rsd/track_2a/predictions/M1_admin_{lang}.jsonl.jsonl"},
        "B0 baseline": {"test": "polyjudge-swissgov-rsd/track_2a/predictions/B0_admin_{lang}.jsonl.jsonl"},
        "M0": {"test": "polyjudge-swissgov-rsd/track_2a/predictions/extra/M0_admin_{lang}.jsonl.jsonl"},
        "M2": {"test": "polyjudge-swissgov-rsd/track_2a/predictions/extra/M2_admin_{lang}.jsonl.jsonl"},
        "M3": {"test": "polyjudge-swissgov-rsd/track_2a/predictions/extra/M3_admin_{lang}.jsonl.jsonl"},
    },
    "RoestiBridge": {
        "submission (70B, n=3)": {"full": "RoestiBridge/track_2a/predictions/submission/RoestiBridge_Apertus-70B_admin_{lang}.jsonl.jsonl"},
        "70B, n=1": {"test": "RoestiBridge/track_2a/predictions/test_70b_n1/predictions_{lang}.jsonl"},
        "70B, n=3 (run folders)": {
            "dev": "RoestiBridge/track_2a/predictions/dev_70b_n3_full/predictions_{lang}.jsonl",
            "test": "RoestiBridge/track_2a/predictions/test_70b_n3/predictions_{lang}.jsonl",
        },
    },
    "rsd-swissgov": {
        "predictions.jsonl": {"full": "rsd-swissgov/track_2a/data/predictions.jsonl"},  # all languages in one file
        "diffalign L15": {"full": "rsd-swissgov/track_2a/src/python/dataset/judging/aperdiff_diffalign_L15_sum_admin_{lang}.jsonl"},
        "diffalign L14-17": {"full": "rsd-swissgov/track_2a/src/python/dataset/judging/aperdiff_diffalign_L14-15-16-17_sum_admin_{lang}.jsonl"},
        "probe L14-17 GRU": {"full": "rsd-swissgov/track_2a/src/python/dataset/judging/aperdiff_probe_L14-15-16-17_sum_gru_devsel_admin_{lang}.jsonl"},
    },
    "SpanDiff-HackApertus": {
        "SpanDiff (dev out-of-fold)": {
            "dev": "SpanDiff-HackApertus/track_2a/predictions/dev/SpanDiff_admin_{lang}.jsonl.jsonl",
            "test": "SpanDiff-HackApertus/track_2a/predictions/test/SpanDiff_admin_{lang}.jsonl.jsonl",
        },
        "SpanDiff full-set file": {"full": "SpanDiff-HackApertus/track_2a/predictions/full/SpanDiff_admin_{lang}.jsonl.jsonl"},
        "ablation L8": {"dev": "SpanDiff-HackApertus/track_2a/predictions/ablation_l8/SpanDiff_admin_{lang}.jsonl.jsonl"},
        "ablation L15": {"dev": "SpanDiff-HackApertus/track_2a/predictions/l15/SpanDiff_admin_{lang}.jsonl.jsonl"},
        "baseline final layer": {"dev": "SpanDiff-HackApertus/track_2a/predictions/baseline_final_layer/SpanDiff_admin_{lang}.jsonl.jsonl"},
    },
}
# plain-language description of each team's best-scoring run (checked against README/report and submitted code)
SYSTEMS = {
    "rsd-swissgov": "Frozen Apertus-v1.5-8B reads each document with the other placed before it; word-level hidden states "
                    "of layers 14-17 are averaged and scored by a small trained GRU probe (about 4.5M parameters). Trained on "
                    "dev labels, so the dev score is in-sample; by default the code picks the training epoch on the test "
                    "split, and the setting used for the submitted model is not recorded.",
    "apertus-swissgov-rsd": "Frozen Apertus-v1.5-8B as a reader: per word, DiffAlign dissimilarity at layers 12, 14, 16 and 18 "
                            "plus contrastive surprisal (how much seeing the other document makes the word more predictable), "
                            "combined by a logistic-regression head trained on dev labels.",
    "hack-apertus-tsemin": "Apertus-v1.5-8B via API, no training: words are ranked by how surprising the model still finds "
                           "them after reading the other version; blocks the model judges to have no counterpart go to the top "
                           "when surprisal agrees. Four fusion settings were tuned on 60 dev pairs.",
    "SpanDiff-HackApertus": "DiffAlign on Apertus-v1.5-8B hidden state 15 with refinements (mean-centring, hubness correction, "
                            "sentence-constrained alignment), combined with surprisal, number-mismatch and simple word features "
                            "by a ridge regression trained on dev labels (dev scores out-of-fold).",
    "polyjudge-swissgov-rsd": "Rule-based sentence alignment marks words in sentences without a counterpart; Apertus-v1.5-70B "
                              "verifies a few extra rule-based flags (numbers, untranslated text). The model changes only 74 "
                              "of 150k test tokens, so the score matches the model-free baseline (0.273).",
    "Hack-Apertus-Swiss-open-source-AI": "Unsupervised DiffAlign on Apertus-v1.5-8B: a word scores 1 minus its best cosine "
                                         "match in the other document, using layer-20 hidden states centred on the document "
                                         "mean. Layer and centring were chosen on dev labels.",
    "RoestiBridge": "Apertus-v1.5-70B is prompted to list meaning differences as excerpts with a 1-5 severity; excerpts are "
                    "mapped to words, averaged over 3 samples and combined with a rule-based number-mismatch score. No "
                    "training; scoring rules were set on a 40-document dev subset.",
    "clauseshift": "DiffAlign with the mmBERT-base encoder (not Apertus) scores all words; Apertus-v1.5-8B re-rates at most 12 "
                   "words per side in 18 documents per language, blended in with weight 0.35. No training.",
}

NOT_SCORED = {
    "hackapertus_submission_track_2_uzh_tk_gm_clean": "no prediction files committed",
    "swissgov-rsd-apertus": "no prediction files committed (skeleton + methodology only)",
    "yt-uzh": "only 10 span-level rows (results/result_on_test.jsonl), not a full prediction set",
}
FIELDS = ["id", "text_a", "text_b", "labels_a", "labels_b"]


def read_rows(source, lang):
    if isinstance(source, tuple):
        with zipfile.ZipFile(SUB / source[0]) as z:
            text = z.read(source[1].format(lang=lang)).decode()
    else:
        text = (SUB / source.format(lang=lang)).read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if r["id"].startswith(f"admin_{lang}_")]


def import_run(team, run, sources):
    """validate and copy one run; returns the dir holding its {full,dev,test}/ files"""
    run_dir = DEST / team / "".join(c if c.isalnum() or c in "-." else "_" for c in run)
    for split, source in sources.items():
        for lang in LANGS:
            rows = read_rows(source, lang)
            gold = [json.loads(line) for line in open(GOLD / split / f"gold_admin_{lang}.jsonl")]
            assert len(rows) == len(gold), f"{team} {split} {lang}: {len(rows)} rows, gold has {len(gold)}"
            for r, g in zip(rows, gold):
                assert r["id"] == g["id"], f"{team} {split} {lang}: id {r['id']} != gold {g['id']}"
                for side in "ab":
                    assert r[f"text_{side}"].split() == g[f"text_{side}"].split(), f"{team} {r['id']}: text_{side} differs"
                    assert len(r[f"labels_{side}"]) == len(g[f"text_{side}"].split()), f"{team} {r['id']}: labels_{side} length"
            out = run_dir / split / f"{team}_admin_{lang}.jsonl.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n" for r in rows))
    return run_dir


def avg(vals):
    return sum(vals) / 3 if vals else None


def cells(scores, sep, none="–"):
    out = []
    for split in ["dev", "test"]:
        vals = scores[split]
        out += [f"{v:.3f}" for v in vals] + [f"{avg(vals):.3f}"] if vals else [none] * 4
    return sep.join(out)


def score_run(team, run, sources):
    run_dir = import_run(team, run, sources)
    scores = {}
    for split in ["dev", "test"]:
        src = split if split in sources else "full" if "full" in sources else None
        scores[split] = spearman(str(run_dir / src / f"{team}_admin_"), split) if src else None
    print(f"{team} | {run}: {scores}")
    return scores


def latex_table(header_col, rows, caption, label, col_spec=None):
    """rows: [(label cells..., scores)]; header_col: leading column names; col_spec: their LaTeX column types"""
    k = len(header_col)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{" + (col_spec or "l" * k) + "c" * 8 + "}", r"\toprule",
           " & " * k + r"\multicolumn{4}{c}{Dev} & \multicolumn{4}{c}{Test} \\",
           rf"\cmidrule(lr){{{k + 1}-{k + 4}}} \cmidrule(lr){{{k + 5}-{k + 8}}}",
           " & ".join(header_col) + r" & DE & FR & IT & Avg & DE & FR & IT & Avg \\", r"\midrule"]
    for *labels, scores in rows:
        tex.append(" & ".join(l.replace("_", r"\_") for l in labels) + " & " + cells(scores, " & ", "--") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return tex


def main():
    runs = {team: {run: score_run(team, run, sources) for run, sources in team_runs.items()} for team, team_runs in TEAMS.items()}
    official = sorted(((team, next(iter(r.items()))) for team, r in runs.items()), key=lambda x: -avg(x[1][1]["test"]))
    # highest per team by test average among runs with test predictions; selected on test -> optimistic
    best = sorted(((team, max(((k, v) for k, v in r.items() if v["test"]), key=lambda kv: avg(kv[1]["test"])))
                   for team, r in runs.items()),
                  key=lambda x: -avg(x[1][1]["test"]))

    head = "| {} | Dev DE | Dev FR | Dev IT | Dev avg | Test DE | Test FR | Test IT | Test avg |"
    sep = "|---" * 9 + "|"
    md = ["Spearman correlation (token-level, SwissGov-RSD, official evaluator). Dev: – = no dev predictions. "
          "Commits: submissions.tsv.", "", "## Official submission per team (sorted by test avg)", "",
          head.format("Team"), sep]
    md += [f"| {team} | {cells(s, ' | ')} |" for team, (run, s) in official]
    md += ["", "## Highest-scoring committed run per team (selected by test avg, so optimistic)", "",
           head.format("Team | Run | System"), sep + "---|---|"]
    md += [f"| {team} | {run}{' (official)' if run == next(iter(runs[team])) else ''} | {SYSTEMS[team]} | {cells(s, ' | ')} |"
           for team, (run, s) in best]
    md += ["", "## All scored runs", "", head.format("Team | Run"), sep + "---|"]
    md += [f"| {team} | {run} | {cells(s, ' | ')} |" for team, r in runs.items() for run, s in r.items()]
    md += ["", "## Not scored"] + [f"- {t}: {why}" for t, why in NOT_SCORED.items()]
    (OUT / "submissions_results_markdown.txt").write_text("\n".join(md) + "\n")

    tex = latex_table(["Team"], [(t, s) for t, (_, s) in official],
                      "Spearman correlation of the official Hack Apertus Track 2A submissions on SwissGov-RSD, computed "
                      "with the official evaluation script from each team's submitted predictions. -- = no dev predictions submitted.",
                      "tab:submissions-official")
    tex += [""] + latex_table(["Team"], [(t, s) for t, (run, s) in best],
                              "Highest-scoring committed prediction set per team, selected by test average (optimistic, "
                              "since the selection uses the test split). -- = no dev predictions.",
                              "tab:submissions-best")
    (OUT / "submissions_results_latex.txt").write_text("\n".join(tex) + "\n")

    systems = [r"\begin{table*}[t]", r"\centering", r"\small",
               r"\begin{tabular}{lp{0.72\textwidth}}", r"\toprule", r"Team & System \\", r"\midrule"]
    systems += [team.replace("_", r"\_") + " & " + SYSTEMS[team] + r" \\" for team, _ in best]
    systems += [r"\bottomrule", r"\end{tabular}",
                r"\caption{System descriptions of the best-scoring run per team (Table~\ref{tab:submissions-best}), "
                r"checked against each team's report and submitted code. Dev-tuned settings make the corresponding dev "
                r"scores optimistic; test scores are unaffected.}",
                r"\label{tab:submissions-systems}", r"\end{table*}"]
    (OUT / "submissions_systems_latex.txt").write_text("\n".join(systems) + "\n")
    print("wrote", OUT / "submissions_results_markdown.txt", OUT / "submissions_results_latex.txt",
          OUT / "submissions_systems_latex.txt")


if __name__ == "__main__":
    main()
