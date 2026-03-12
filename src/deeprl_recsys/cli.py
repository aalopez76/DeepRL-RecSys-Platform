"""CLI entry-point -- Typer application.

Delegates all logic to the SDK (core/) and pipeline orchestrators
(pipelines/).  Each command loads config via :func:`core.config.load_config`
and delegates to the corresponding pipeline function.

Usage::

    deeprl-recsys prepare --config configs/experiments/exp1_dqn_movielens.yaml
    deeprl-recsys train   --config configs/experiments/exp1_dqn_movielens.yaml
    deeprl-recsys evaluate --config configs/experiments/exp1_dqn_movielens.yaml
    deeprl-recsys export  --config configs/experiments/exp1_dqn_movielens.yaml
    deeprl-recsys serve   --artifact artifacts/models/latest
    deeprl-recsys list-plugins
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(
    name="deeprl-recsys",
    help="DeepRL-RecSys-Platform CLI - reproducible RL-based recommendations.",
    add_completion=False,
)


def _echo(msg: str) -> None:
    """Print a message to stdout (Windows-safe)."""
    typer.echo(msg)


def _set_verbose(verbose: bool) -> None:
    """Set logging level to DEBUG if --verbose is passed."""
    if verbose:
        logging.getLogger("deeprl_recsys").setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")


def _load_config(config_path: Path, overrides: dict | None = None) -> dict:
    """Load and resolve config from YAML file.

    Uses ``core.config.load_config`` when a real YAML is given,
    otherwise returns the overrides dict.
    """
    from deeprl_recsys.core.config import load_config

    cfg = load_config(
        default_path=None,
        experiment_path=config_path,
        overrides=overrides or {},
    )
    return cfg.model_dump()


@app.command()
def prepare(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment config YAML"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip write operations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Prepare and validate a dataset."""
    from pipelines.prepare_data import run_prepare

    _set_verbose(verbose)
    _echo(f">> prepare  config={config}")

    try:
        cfg = _load_config(config)
        result = run_prepare(cfg, dry_run=dry_run)

        if result["is_valid"]:
            _echo(
                f"[OK] Dataset valid - {result['n_rows']} rows -> "
                f"{result['output_path']}"
            )
        else:
            _echo("[FAIL] Validation failed:")
            for e in result["errors"]:
                _echo(f"  - {e}")
            raise typer.Exit(code=1)

        if result["warnings"]:
            for w in result["warnings"]:
                _echo(f"  [WARN] {w}")

    except typer.Exit:
        raise
    except Exception as exc:
        _echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment config YAML"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip checkpoint writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Train an agent."""
    from pipelines.train import run_train

    _set_verbose(verbose)
    _echo(f">> train  config={config}")

    try:
        cfg = _load_config(config)
        result = run_train(cfg, dry_run=dry_run)

        _echo("Training Complete:")
        _echo(f"  Agent : {result['agent_name']}")
        _echo(f"  Steps : {result['steps_completed']}")
        _echo(f"  Model : {result['model_path']}")

    except Exception as exc:
        _echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def evaluate(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment config YAML"
    ),
    fail_on: Optional[str] = typer.Option(
        None, "--fail-on", help="Exit 1 if verdict severity matches (e.g. 'error')"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip report writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run OPE evaluation with diagnostics."""
    from pipelines.evaluate import run_evaluate

    _set_verbose(verbose)
    _echo(f">> evaluate  config={config}")

    try:
        cfg = _load_config(config)
        if fail_on:
            cfg.setdefault("ope", {})["fail_on"] = fail_on

        result = run_evaluate(cfg, dry_run=dry_run)

        # Estimates
        _echo("OPE Estimates:")
        for name, value in result["estimates"].items():
            _echo(f"  {name:>10s} = {value:.6f}")

        # Verdict
        verdict = result["verdict"]
        sev = verdict["severity"].upper()
        _echo(f"\nVerdict: [{sev}]")
        for w in verdict["warnings"]:
            _echo(f"  [WARN] {w}")

    except SystemExit as exc:
        raise typer.Exit(code=exc.code) from exc
    except Exception as exc:
        _echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def export(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment config YAML"
    ),
    output: Path = typer.Option(
        "artifacts/models/latest", "--output", "-o", help="Output directory"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip artifact writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Export a canonical artifact."""
    from pipelines.export import run_export

    _set_verbose(verbose)
    _echo(f">> export  config={config}, output={output}")

    try:
        cfg = _load_config(config)
        cfg.setdefault("export", {})["output_dir"] = str(output)

        artifact_path = run_export(cfg, dry_run=dry_run)
        _echo(f"[OK] Artifact exported -> {artifact_path}")

    except Exception as exc:
        _echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def serve(
    artifact: Path = typer.Option(
        ..., "--artifact", "-a", help="Path to artifact directory"
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Launch the FastAPI serving endpoint."""
    from pipelines.serve import run_serve

    _set_verbose(verbose)
    _echo(f">> serve  artifact={artifact}, {host}:{port}")
    run_serve(str(artifact), host=host, port=port)


@app.command()
def ui() -> None:
    """Launch the interactive Streamlit Dashboard."""
    import subprocess
    import importlib.util

    _echo(">> starting  deeprl-recsys ui")

    if importlib.util.find_spec("streamlit") is None:
        _echo("[ERROR] Streamlit no está instalado. Ejecuta: pip install -e .[ui]")
        raise typer.Exit(code=1)

    # Resolve app.py file location dynamically
    base_dir = Path(__file__).resolve().parent
    app_path = base_dir / "ui" / "app.py"

    if not app_path.exists():
        _echo(f"[ERROR] No se encontró la interfaz en {app_path}")
        raise typer.Exit(code=1)

    try:
        _echo("Iniciando Streamlit...")
        subprocess.run(
            ["streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        _echo("\n[OK] Dashboard cerrado.")
    except Exception as exc:
        _echo(f"[ERROR] Falló al lanzar streamlit: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="list-plugins")
def list_plugins() -> None:
    """List registered and discovered plugins."""
    from deeprl_recsys.core.registry import list_registered, load_plugins

    _echo(">> list-plugins\n")
    load_plugins()

    _echo("Registered Plugins:")
    for category in ("agents", "environments", "simulators", "estimators", "metrics"):
        names = list_registered(category)
        _echo(f"  {category:>12s} : {', '.join(names) if names else '(none)'}")


if __name__ == "__main__":
    app()
