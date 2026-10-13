"""MLSeal command-line interface: one command per pipeline stage, each a CI gate."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Secure MLOps pipeline for a phishing-website classifier.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CONFIG = Path("configs/pipeline.yaml")
BUNDLE = Path("build/bundle")
KEYS = Path("runtime/keys")
BASELINE = Path("reports/metrics.json")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Pipeline configuration.")]
BundleOpt = Annotated[Path, typer.Option("--bundle", "-b", help="Release bundle directory.")]


def _fail(message: str) -> typer.Exit:
    console.print(f"[bold red]FAILED[/bold red] {message}")
    return typer.Exit(code=1)


def _passphrase() -> bytes | None:
    # From the environment only: a CLI argument would leak into shell history and `ps`.
    value = os.environ.get("MLSEAL_KEY_PASSPHRASE")
    return value.encode() if value else None


def _checks_table(title: str, rows: list[tuple[str, bool, str]]) -> Table:
    table = Table(title=title, title_justify="left")
    for col in ("check", "result", "detail"):
        table.add_column(col)
    for name, passed, detail in rows:
        table.add_row(name, "[green]pass[/green]" if passed else "[red]FAIL[/red]", detail)
    return table


@app.command()
def validate(
    config: ConfigOpt = CONFIG,
    out: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path(
        "build/data_validation.json"
    ),
    write_profile: Annotated[
        bool, typer.Option(help="Regenerate the reference profile from this data (review it!).")
    ] = False,
) -> None:
    """Data gate: hash pin, schema, allowed values, balance, duplicates, conflicts, drift."""
    from mlseal.config import load_config
    from mlseal.data import read_arff
    from mlseal.validation import load_reference, profile, validate_file

    cfg = load_config(config)
    if write_profile:
        prof = profile(read_arff(cfg.data.path).X)
        cfg.data.reference_profile.write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
        console.print(f"Reference profile written to {cfg.data.reference_profile}")
    report = validate_file(
        cfg.data.path, cfg.data.policy, load_reference(cfg.data.reference_profile)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    console.print(
        _checks_table(
            f"Data gate: {cfg.data.path}", [(c.name, c.passed, c.detail) for c in report.checks]
        )
    )
    if not report.passed:
        raise _fail("data validation; nothing will be trained on this file")


@app.command("data-provenance")
def data_provenance(config: ConfigOpt = CONFIG) -> None:
    """Download the upstream archive and check its member against the pinned digest."""
    from mlseal.config import load_config
    from mlseal.data import archive_member_digest, download, sha256_bytes

    cfg = load_config(config)
    up = cfg.data.upstream
    if up is None:
        raise _fail("no upstream section in the configuration")
    archive = download(up.url)
    member = archive_member_digest(archive, up.member)
    same_archive = sha256_bytes(archive) == up.archive_sha256
    console.print(f"archive {up.url}: {'unchanged' if same_archive else 'rebuilt upstream'}")
    if member != cfg.data.policy.sha256:
        raise _fail(f"{up.member} upstream is {member}, pinned {cfg.data.policy.sha256}")
    console.print(f"[green]ok[/green] {up.member} matches the pinned digest {member}")


@app.command()
def train(config: ConfigOpt = CONFIG, bundle: BundleOpt = BUNDLE) -> None:
    """Validate, de-duplicate, select by CV, evaluate on the hold-out, export model.json."""
    from mlseal.config import load_config
    from mlseal.pipeline import train as run
    from mlseal.validation import DataValidationError

    try:
        run(load_config(config), bundle, config_path=config, log=console.print)
    except DataValidationError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"Bundle written to [bold]{bundle}[/bold]")


@app.command()
def gate(
    config: ConfigOpt = CONFIG,
    bundle: BundleOpt = BUNDLE,
    baseline: Annotated[Path, typer.Option(help="Metrics of the approved model.")] = BASELINE,
    expect_reproducible: Annotated[
        bool, typer.Option(help="Also require the committed report to be reproduced exactly.")
    ] = False,
) -> None:
    """Quality gates: metric floors, maximum regression vs the baseline, export parity."""
    from mlseal.config import load_config
    from mlseal.gates import run_gates, summary

    cfg = load_config(config)
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    base = json.loads(baseline.read_text(encoding="utf-8")) if baseline.is_file() else None
    results = run_gates(metrics, cfg.gates, base, expect_reproducible=expect_reproducible)
    report = summary(results, str(baseline) if base else None)
    (bundle / "gates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(_checks_table("Quality gates", [(r.name, r.passed, r.detail) for r in results]))
    if not report["passed"]:
        raise _fail("quality gates; this model must not be released")


@app.command()
def scan(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories to scan.")],
    forbid_pickle: Annotated[
        bool, typer.Option(help="Fail on any pickle, even one with only allowlisted globals.")
    ] = False,
) -> None:
    """Static pickle scanner: flags non-allowlisted imports without ever unpickling."""
    from mlseal.scanner import scan_path

    reports = [r for p in paths for r in scan_path(p)]
    if not reports:
        raise _fail("no files to scan")  # an empty scan must not pass as a clean one
    table = Table(title="Model file scan", title_justify="left")
    for col in ("file", "kind", "verdict", "findings"):
        table.add_column(col)
    bad = 0
    for r in reports:
        refused = not r.safe or (forbid_pickle and r.is_pickle)
        bad += refused
        findings = "; ".join(f"{f.severity.name}: {f.message}" for f in r.findings[:3])
        if forbid_pickle and r.is_pickle and r.safe:
            findings = "pickle not allowed in a release bundle"
        verdict = "[red]REFUSED[/red]" if refused else "[green]ok[/green]"
        table.add_row(r.path, r.kind, verdict, findings or "-")
    console.print(table)
    if bad:
        raise _fail(f"{bad} of {len(reports)} files refused")
    console.print(f"{len(reports)} files scanned, none refused")


@app.command()
def keygen(
    out_dir: Annotated[Path, typer.Option(help="Directory for signing.key/.pub.")] = KEYS,
) -> None:
    """Create an Ed25519 signing key pair (private key 0600, encrypted if a passphrase is set)."""
    from mlseal.signing import generate_key, key_id, save_keypair

    key = generate_key()
    try:
        private, public = save_keypair(key, out_dir, _passphrase())
    except FileExistsError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"Private key: {private} (keep it out of git)\nPublic key:  {public}")
    console.print(f"Key id: {key_id(key.public_key())}")


@app.command()
def sign(
    bundle: BundleOpt = BUNDLE,
    key: Annotated[Path, typer.Option(help="Ed25519 private key (PEM).")] = KEYS / "signing.key",
) -> None:
    """Sign the bundle manifest, but only if every gate passed and the bundle holds no pickle."""
    from mlseal import modelfile
    from mlseal.provenance import BUNDLE_FILES, ProvenanceError, build_statement
    from mlseal.scanner import scan_path
    from mlseal.signing import load_private_key, sign_bundle

    for name, label in (("data_validation.json", "data gate"), ("gates.json", "quality gates")):
        path = bundle / name
        if not path.is_file() or not json.loads(path.read_text(encoding="utf-8"))["passed"]:
            raise _fail(f"{label} did not pass; refusing to sign")
    pickles = [r.path for r in scan_path(bundle) if r.is_pickle]
    if pickles:
        raise _fail(f"pickles, or files the scanner refused, in the bundle: {pickles}")
    modelfile.loads((bundle / "model.json").read_bytes())  # never sign an unloadable model
    try:
        statement = build_statement(bundle)
    except ProvenanceError as exc:
        raise _fail(str(exc)) from exc
    envelope = sign_bundle(bundle, statement, load_private_key(key, _passphrase()))
    console.print(f"Signed {len(BUNDLE_FILES)} files -> [bold]{envelope}[/bold]")


@app.command()
def verify(
    bundle: BundleOpt = BUNDLE,
    public_key: Annotated[
        list[Path] | None,
        typer.Option(help="Trusted public key(s), PEM. [default: runtime/keys/signing.pub]"),
    ] = None,
) -> None:
    """Verify the signature and every file digest, then load the model strictly."""
    from mlseal import modelfile
    from mlseal.signing import VerificationError, load_public_key, verify_bundle

    keys = public_key or [KEYS / "signing.pub"]
    try:
        verified = verify_bundle(bundle, [load_public_key(p.read_bytes()) for p in keys])
        model = modelfile.loads(verified.files["model.json"])
    except (VerificationError, modelfile.ModelFormatError) as exc:
        raise _fail(str(exc)) from exc
    pred = verified.statement["predicate"]
    table = Table(title="Verified release", title_justify="left", show_header=False)
    table.add_row("model", f"{model.name} {model.version} ({model.kind})")
    table.add_row("signed by", verified.keyid)
    table.add_row("commit", f"{pred['source']['commit']} (dirty: {pred['source']['dirty']})")
    table.add_row("builder", pred["builder"].get("run", pred["builder"]["id"]))
    for subject in verified.statement["subject"]:
        table.add_row(subject["name"], f"sha256:{subject['digest']['sha256'][:16]}… ok")
    console.print(table)


@app.command()
def promote(
    bundle: BundleOpt = BUNDLE,
    reports: Annotated[Path, typer.Option(help="Committed reports directory.")] = Path("reports"),
) -> None:
    """Make this run the new baseline: copy its report files and render the figures."""
    from mlseal.plots import render_all

    gates = json.loads((bundle / "gates.json").read_text(encoding="utf-8"))
    if not gates["passed"]:
        raise _fail("only a bundle that passed the quality gates can be promoted")
    reports.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        ("metrics.json", "metrics.json"),
        ("model_card.md", "MODEL_CARD.md"),
        ("data_validation.json", "data_validation.json"),
        ("gates.json", "gates.json"),
    ):
        shutil.copyfile(bundle / src, reports / dst)
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    for fig in render_all(metrics, reports / "figures"):
        console.print(f"Figure: {fig}")
    console.print(f"Baseline updated in [bold]{reports}[/bold]; review and commit it.")


@app.command()
def serve(
    bundle: Annotated[Path | None, typer.Option(help="Overrides MLSEAL_BUNDLE.")] = None,
    public_key: Annotated[Path | None, typer.Option(help="Overrides MLSEAL_PUBLIC_KEY.")] = None,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8080,
) -> None:
    """Verify the signed bundle and serve it (refuses to start on any verification failure)."""
    import dataclasses

    import uvicorn

    from mlseal.serving import Settings, create_app
    from mlseal.serving.logs import configure

    settings = Settings.from_env()
    if bundle is not None:
        settings = dataclasses.replace(settings, bundle=bundle)
    if public_key is not None:
        settings = dataclasses.replace(settings, public_keys=(public_key,))
    configure()
    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        log_config=None,
        access_log=False,  # replaced by the app's structured access log (no client IPs)
        server_header=False,
        proxy_headers=False,
        limit_concurrency=64,
        timeout_keep_alive=5,
        h11_max_incomplete_event_size=16 * 1024,
    )


if __name__ == "__main__":
    app()
