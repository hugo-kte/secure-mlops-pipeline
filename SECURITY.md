# Security policy

## Reporting a vulnerability

Please do not describe a vulnerability in a public issue. GitHub private vulnerability
reporting is not enabled on this repository: open an issue titled "Security contact",
without any details, and the maintainer will reply with a private channel.

In scope: the code in this repository, in particular any way to get past one of its
controls. Examples: a bundle that `mlseal verify` or the server accepts although it was
modified or signed by an untrusted key, a pickle the scanner passes that runs code when
loaded, a request that crashes the server or reaches the logs, a workflow change that
weakens a CI gate. Out of scope: the classifier's accuracy and its robustness to evasion
(documented as not evaluated), and vulnerabilities in third-party packages (report them
upstream; how this repository picks up their fixes is described below).

Only the latest commit on `main` is maintained.

## How vulnerable dependencies are handled

* **Detection.** pip-audit checks every locked package against the PyPI and OSV advisory
  databases: the serving dependencies, but also the training stack and the dev and audit
  tools, because they all run in CI next to the signing step. It runs on every push and
  pull request and once a week on `main`, so a new advisory fails the build even when
  nothing is pushed. A failing job blocks the release: the provenance attestation is only
  produced when every job has passed.
* **Cooldown.** uv resolves nothing uploaded in the 14 days before the lock was cut
  (`exclude-newer` in [`pyproject.toml`](pyproject.toml)), so a hijacked release has time
  to be detected and removed before it can enter `uv.lock`. The date moves forward only in
  a reviewed `make lock` commit.
* **Exceptions.** A security fix often needs a release younger than 14 days. A package may
  skip the cooldown only under this rule:
  1. *What qualifies.* The release is the smallest one that fixes every advisory pip-audit
     reported for that package (the fixed version stated by the advisories), and its PyPI
     files carry a publish attestation (PEP 740) from the project's own release workflow.
     The only other kind is a reviewed version pin of a dev-only tool that is never shipped.
  2. *How narrow.* The exception ends at the UTC midnight after that release's last
     upload: it admits that release and nothing published after it.
  3. *Where it is recorded.* `[tool.uv.exclude-newer-package]` in `pyproject.toml`, with
     the advisories, the admitted version and the attestation checked in
     `[tool.mlseal.cooldown-exceptions]`, in the same reviewed commit as the lock update.
  4. *When it ends.* It is deleted once the cooldown date moves past it.

  [`tests/test_policies.py`](tests/test_policies.py) enforces points 2 to 4 (every
  exception is justified, admits exactly the locked release, and has not expired), and
  checks that the rule catches each kind of breach.

## Advisory log

**2026-10-29: 56 advisories in four locked build tools.** CI on `main` was re-run on a
commit that had passed every gate on 2026-10-13
([run 38407015641, attempt 2](https://github.com/hugo-kte/secure-mlops-pipeline/actions/runs/38407015641)).
The supply-chain job failed: advisories published between 2026-10-19 and 2026-10-28 hit four
locked packages. pip-audit counted 56 findings (55 distinct IDs; one is listed with two
fixed versions). The attestation job was skipped, as it is whenever a gate fails. None of
the four packages is installed in the serving image.

| Package | Locked | Advisories | Fixed release (uploaded) | Pulled in by |
|---|---|---|---|---|
| certifi | 2026.7.22 | 1: emSign roots with compromised intermediate CA keys ([GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2)) | 2026.10.19 (2026-10-20) | requests, for pip-audit's TLS connections to the advisory services |
| filelock | 4.0.6 | 14: lock-file races open to another local user, such as symlink swaps and forged markers (e.g. [GHSA-v4x8-3wr5-3wq8](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8)) | 4.2.1 (2026-10-17) | cachecontrol, pip-audit's HTTP cache |
| msgpack | 1.2.2 | 1: memory leak on map-key validation failure ([GHSA-j586-36cw-2gc2](https://github.com/advisories/GHSA-j586-36cw-2gc2)) | 1.2.3 (2026-09-29) | cachecontrol, pip-audit's HTTP cache |
| pillow | 12.3.0 | 39: image decoder and PDF parser flaws, including a heap overflow in the bundled OpenJPEG ([GHSA-78g4-xqwv-q3g8](https://github.com/advisories/GHSA-78g4-xqwv-q3g8)) | 13.0.0 (2026-10-16) | matplotlib, for the report figures |

Resolution: each package moved to the smallest release that clears its advisories, under
four security exceptions (msgpack 1.2.3 is a month old, but this lock's cooldown date,
2026-09-29T00:00Z, excludes it by two and a half hours). certifi 2026.10.18 was skipped: it
removed every emSign root, PYSEC-2026-4320 still lists it as affected, and the advisory
recommends 2026.10.19. All four releases carry PyPI publish attestations from their
projects' release workflows. Nothing else in the lock moved, the image requirements did not
change, and the reproduction gate retrained the model under the new lock and matched every
value of the committed report.
