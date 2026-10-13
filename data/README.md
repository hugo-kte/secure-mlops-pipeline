# Data: UCI Phishing Websites

`phishing_websites.arff` is the file `Training Dataset.arff` from the UCI Machine Learning
Repository archive, **byte for byte**. The only change is the file name.

| | |
|---|---|
| Dataset | Phishing Websites (UCI ML Repository, id 327) |
| Authors | Rami Mohammad, Lee McCluskey (with Fadi Thabtah) |
| DOI | [10.24432/C51W2X](https://doi.org/10.24432/C51W2X) |
| Archive | <https://archive.ics.uci.edu/static/public/327/phishing+websites.zip> |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) |
| Content | 11,055 rows, 30 ternary features + label `Result` (-1 phishing, 1 legitimate) |
| SHA-256 of this file | `3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6` |
| SHA-256 of the archive (downloaded 2026-10-13) | `514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940` |

The file digest is pinned in [`configs/pipeline.yaml`](../configs/pipeline.yaml) and checked
by the data gate before every training run. `make data-provenance` downloads the archive
again and checks that its `Training Dataset.arff` member still has this digest (the archive
itself may be rebuilt by UCI, so its own digest is informational).

Feature semantics follow the authors' extraction rules, documented in the "Phishing Websites
Features" note shipped in the archive. The label encoding (-1 = phishing) is the one the
strongest indicators agree with: for example 98.9% of rows with `URL_of_Anchor = -1` and
85.8% of rows with `SSLfinal_State = -1` are labelled -1.

## Citation

> Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C51W2X

```bibtex
@misc{phishing_websites_327,
  author       = {Mohammad, Rami and McCluskey, Lee},
  title        = {{Phishing Websites}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {DOI: https://doi.org/10.24432/C51W2X}
}
```

Related paper: R. M. Mohammad, F. Thabtah and L. McCluskey, "An assessment of features
related to phishing websites using an automated technique", *International Conference for
Internet Technology and Secured Transactions*, 2012.
