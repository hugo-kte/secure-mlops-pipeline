"""Model card (Mitchell et al., 2019) generated from metrics.json: no hand-copied numbers."""

from __future__ import annotations

from typing import Any

LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (phishing caught)",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
}
DECIMALS = {"roc_auc": 4, "average_precision": 4}
PROTOCOLS = {
    "naive": "Row-level split (duplicates straddle train and test)",
    "grouped": "Grouped split (identical vectors kept together)",
    "deduplicated": "De-duplicated (one row per distinct vector)",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(metric: str, value: float, ci: list[float]) -> str:
    d = DECIMALS.get(metric, 3)
    return f"{value:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def render_model_card(metrics: dict[str, Any], validation: dict[str, Any]) -> str:
    m, d, t = metrics["model"], metrics["data"], metrics["test"]
    proto, sel = metrics["protocol"], metrics["selection"]
    effect, conf, mats = metrics["duplicates_effect"], t["confusion"], metrics["materials"]

    rows = "\n".join(f"| {LABELS[k]} | {_ci(k, t['metrics'][k], t['ci95'][k])} |" for k in LABELS)

    def mark(i: int, c: dict[str, Any]) -> str:
        tags = (["**selected**"] if i == sel["selected"] else []) + (
            ["best mean"] if c.get("best_mean") else []
        )
        return f" ({', '.join(tags)})" if tags else ""

    cand_rows = "\n".join(
        f"| {c['description']}{mark(i, c)} "
        f"| {c['cv']['roc_auc']['mean']:.4f} ± "
        f"{c['cv']['roc_auc']['ci95'][1] - c['cv']['roc_auc']['mean']:.4f} "
        f"| {c['cv']['f1']['mean']:.4f} |"
        for i, c in enumerate(sel["candidates"])
    )
    effect_rows = "\n".join(
        f"| {PROTOCOLS[name]} | {r['rows']:,} | {_pct(r['test_rows_seen_in_train'])} "
        f"| {_pct(1 - r['accuracy']['mean'])} | {r['f1']['mean']:.3f} "
        f"| {r['roc_auc']['mean']:.4f} |"
        for name, r in effect.items()
    )
    checks = ", ".join(c["name"] for c in validation["checks"])
    ref = metrics.get("reference_model")
    ref_line = (
        f"For reference, {ref['description']} reaches ROC-AUC {ref['metrics']['roc_auc']:.4f} "
        f"and F1 {ref['metrics']['f1']:.3f} on the same test set.\n"
        if ref
        else ""
    )
    commit = mats["source"]["commit"] or "unknown"
    selection_metric = LABELS.get(sel["metric"], sel["metric"])
    sections = [
        f"# Model card: {m['name']} {m['version']}",
        "## Model details",
        "\n".join(
            [
                f"* **Type:** {m['description']}; binary classifier, positive class = phishing.",
                "* **Format:** `model.json` (`mlseal.model/v1`), plain JSON parameters evaluated"
                " by a numpy runtime. No pickle: loading cannot execute code.",
                f"* **Digest:** `sha256:{m['sha256']}` ({m['bytes']:,} bytes).",
                f"* **Built from:** dataset `sha256:{mats['dataset']['sha256'][:16]}…`, config"
                f" `sha256:{mats['config']['sha256'][:16]}…`, commit `{commit[:12]}`.",
                f"* **Decision threshold:** {proto['threshold']} on the phishing probability.",
            ]
        ),
        "## Intended use",
        "A triage signal for phishing-website detection on the 30 features of the UCI Phishing"
        " Websites dataset (URL, page-content and domain indicators, ternary-encoded). It is a"
        " demonstration model for a secure delivery pipeline: use it to rank or flag pages for"
        " review, not as the sole basis for blocking. It does not extract features from raw"
        " URLs or HTML.",
        "## Training data",
        f"UCI Phishing Websites (Mohammad & McCluskey, 2012; CC BY 4.0), {d['rows']:,} rows."
        f" Only {d['unique_vectors']:,} feature vectors are distinct: {d['duplicate_rows']:,}"
        f" rows ({_pct(d['duplicate_share'])}) repeat an earlier row, and"
        f" {d['conflicting_vectors']} vectors ({d['conflicting_rows']} rows) appear with both"
        " labels. The data is de-duplicated to one row per vector (majority label;"
        f" {d['ties_dropped']} ties dropped): {d['deduplicated_rows']:,} vectors,"
        f" {d['deduplicated_phishing']:,} phishing and {d['deduplicated_legitimate']:,}"
        " legitimate.",
        f"Data gate: passed ({checks}).",
        "## Evaluation",
        f"{proto['evaluation'].capitalize()}: {d['train_rows']:,} training and"
        f" {d['test_rows']:,} test vectors (seed {proto['seed']}), so no test vector was seen"
        " in training. 95% confidence intervals: percentile bootstrap over test vectors"
        f" ({proto['bootstrap_resamples']:,} resamples).",
        f"| Metric | Test value [95% CI] |\n|---|---|\n{rows}",
        f"Confusion matrix at threshold {proto['threshold']}: {conf['tp']} phishing pages"
        f" caught, {conf['fn']} missed, {conf['fp']} legitimate pages flagged, {conf['tn']}"
        " correctly passed.",
        *([ref_line.strip()] if ref_line else []),
        f"**Model selection.** {proto['cv_folds']}-fold stratified CV on the training split"
        f" only, by {selection_metric}; the {sel['rule']} is shipped. ± is the half-width of a"
        " 95% t-interval over folds.",
        f"| Candidate | CV ROC-AUC | CV F1 |\n|---|---|---|\n{cand_rows}",
        f"**Effect of duplicates.** Same model, {proto['cv_folds']}-fold CV, mean over folds:",
        "| Protocol | Rows | Test rows seen in training | Error rate | F1 | ROC-AUC |\n"
        f"|---|---:|---:|---:|---:|---:|\n{effect_rows}",
        "A row-level split puts copies of the same vector on both sides of the split and"
        " overstates performance; the de-duplicated hold-out above is the honest estimate.",
        "## Limitations",
        "\n".join(
            [
                "* The data was collected around 2012-2015; phishing techniques and the web have"
                " changed. Expect lower performance on current traffic and re-validate before"
                " any real use.",
                "* Features are pre-computed by the dataset authors' rules; this repository ships"
                " no feature extractor, so production use needs one that reproduces those rules"
                " exactly.",
                "* The de-duplicated test set weights every distinct vector equally. That is not"
                " the deployment distribution, which is unknown for this dataset.",
                "* Robustness to adaptive attackers (evasion) was not evaluated.",
            ]
        ),
        "## Security and provenance",
        "The release bundle is described by an in-toto statement (subjects: every bundle file"
        " with its SHA-256; predicate: dataset, lock file, config and commit digests, evaluation"
        " and gate outcomes) signed with Ed25519 in a DSSE envelope. The server verifies the"
        " signature and every digest before parsing the model, and refuses to start otherwise."
        " See `docs/THREAT_MODEL.md`.",
    ]
    return "\n\n".join(sections) + "\n"
