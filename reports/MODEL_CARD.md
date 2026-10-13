# Model card: phishing-gbm 1.0.0

## Model details

* **Type:** gradient boosting (300 trees, depth 4, learning rate 0.1); binary classifier, positive class = phishing.
* **Format:** `model.json` (`mlseal.model/v1`), plain JSON parameters evaluated by a numpy runtime. No pickle: loading cannot execute code.
* **Digest:** `sha256:1178d8112842d508a3d4ea93f8e961429c71e88b8aa123718436f7f8bad6b82c` (245,921 bytes).
* **Built from:** dataset `sha256:3bcf93b161230653…`, config `sha256:64b249546cdd2a2d…`, commit `24730e427e35`.
* **Decision threshold:** 0.5 on the phishing probability.

## Intended use

A triage signal for phishing-website detection on the 30 features of the UCI Phishing Websites dataset (URL, page-content and domain indicators, ternary-encoded). It is a demonstration model for a secure delivery pipeline: use it to rank or flag pages for review, not as the sole basis for blocking. It does not extract features from raw URLs or HTML.

## Training data

UCI Phishing Websites (Mohammad & McCluskey, 2012; CC BY 4.0), 11,055 rows. Only 5,785 feature vectors are distinct: 5,270 rows (47.7%) repeat an earlier row, and 64 vectors (357 rows) appear with both labels. The data is de-duplicated to one row per vector (majority label; 13 ties dropped): 5,772 vectors, 2,972 phishing and 2,800 legitimate.

Data gate: passed (integrity, parse, schema, allowed_values, row_count, label_balance, duplicates, label_conflicts, feature_drift, train_test_overlap).

## Evaluation

Stratified hold-out on de-duplicated feature vectors: 4,617 training and 1,155 test vectors (seed 42), so no test vector was seen in training. 95% confidence intervals: percentile bootstrap over test vectors (2,000 resamples).

| Metric | Test value [95% CI] |
|---|---|
| Accuracy | 0.966 [0.956, 0.976] |
| Precision | 0.962 [0.946, 0.976] |
| Recall (phishing caught) | 0.973 [0.960, 0.986] |
| F1 | 0.967 [0.957, 0.977] |
| ROC-AUC | 0.9945 [0.9918, 0.9967] |
| Average precision | 0.9948 [0.9920, 0.9970] |

Confusion matrix at threshold 0.5: 579 phishing pages caught, 16 missed, 23 legitimate pages flagged, 537 correctly passed.

For reference, logistic regression (C=1) reaches ROC-AUC 0.9764 and F1 0.921 on the same test set.

**Model selection.** 5-fold stratified CV on the training split only, by ROC-AUC; the simplest candidate within one standard error of the best CV mean is shipped. ± is the half-width of a 95% t-interval over folds.

| Candidate | CV ROC-AUC | CV F1 |
|---|---|---|
| logistic regression (C=1) | 0.9788 ± 0.0054 | 0.9307 |
| gradient boosting (200 trees, depth 3, learning rate 0.1) | 0.9924 ± 0.0023 | 0.9592 |
| gradient boosting (300 trees, depth 4, learning rate 0.1) (**selected**) | 0.9942 ± 0.0018 | 0.9633 |
| gradient boosting (400 trees, depth 5, learning rate 0.05) (best mean) | 0.9943 ± 0.0014 | 0.9658 |

**Effect of duplicates.** Same model, 5-fold CV, mean over folds:

| Protocol | Rows | Test rows seen in training | Error rate | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Row-level split (duplicates straddle train and test) | 11,055 | 65.4% | 3.4% | 0.961 | 0.9957 |
| Grouped split (identical vectors kept together) | 11,055 | 0.0% | 4.4% | 0.950 | 0.9936 |
| De-duplicated (one row per distinct vector) | 5,772 | 0.0% | 3.7% | 0.964 | 0.9953 |

A row-level split puts copies of the same vector on both sides of the split and overstates performance; the de-duplicated hold-out above is the honest estimate.

## Limitations

* The data was collected around 2012-2015; phishing techniques and the web have changed. Expect lower performance on current traffic and re-validate before any real use.
* Features are pre-computed by the dataset authors' rules; this repository ships no feature extractor, so production use needs one that reproduces those rules exactly.
* The de-duplicated test set weights every distinct vector equally. That is not the deployment distribution, which is unknown for this dataset.
* Robustness to adaptive attackers (evasion) was not evaluated.

## Security and provenance

The release bundle is described by an in-toto statement (subjects: every bundle file with its SHA-256; predicate: dataset, lock file, config and commit digests, evaluation and gate outcomes) signed with Ed25519 in a DSSE envelope. The server verifies the signature and every digest before parsing the model, and refuses to start otherwise. See `docs/THREAT_MODEL.md`.
