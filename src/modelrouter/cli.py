"""`modelrouter` command line: data preparation, training, evaluation and serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare datasets.")
app.add_typer(data_app, name="data")

DATA_DIR = Path("data")


@data_app.command("prepare")
def data_prepare(
    raw_dir: Annotated[Path, typer.Option()] = DATA_DIR / "raw",
    out_dir: Annotated[Path, typer.Option()] = DATA_DIR / "processed",
    download: Annotated[bool, typer.Option(help="Fetch the pinned files first.")] = True,
) -> None:
    """Download RouterBench + GSM8K gold, re-grade GSM8K, write data/processed/routerbench_0shot."""
    from modelrouter.data import routerbench

    if download:
        routerbench.download(raw_dir)
    manifest = routerbench.prepare(raw_dir, out_dir)
    typer.echo(
        json.dumps(
            {
                "rows_out": manifest.rows_out,
                "dropped": manifest.dropped,
                "families": manifest.families,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
