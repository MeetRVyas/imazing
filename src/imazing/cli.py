"""Command-line interface for imazing.

A thin wrapper around the Imazing library for quick, no-code operations
from the terminal -- resize a batch of images, check a photo's stats,
pull text off a screenshot, and so on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from . import Imazing, __version__
from .exceptions import ImazingError

app = typer.Typer(
    name="imazing",
    help="A unified computer-vision toolkit -- quick image operations from the terminal.",
    add_completion=False,
    no_args_is_help=True,
)

logger = logging.getLogger("imazing")

_INPUT_ARG = typer.Argument(..., exists=True, readable=True, help="Input image path.")
_OUTPUT_ARG = typer.Argument(..., help="Where to save the result.")


def _version_callback(value: bool):
    if value:
        typer.echo(f"imazing {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show the installed version and exit.",
    ),
):
    """imazing: a unified computer-vision toolkit."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def _load_or_exit(input_path: Path) -> Imazing:
    try:
        return Imazing(str(input_path))
    except ImazingError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def _save(im: Imazing, output: Path):
    im.save(str(output))
    typer.echo(f"Saved {output}")


@app.command()
def info(input: Path = _INPUT_ARG):
    """Show dimensions, channels, and basic pixel statistics."""
    im = _load_or_exit(input)
    for key, value in im.get_stats().items():
        typer.echo(f"{key}: {value}")


@app.command()
def resize(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    width: Optional[int] = typer.Option(None, help="Target width in pixels."),
    height: Optional[int] = typer.Option(None, help="Target height in pixels."),
):
    """Resize an image. Give one dimension to preserve aspect ratio, or both to set exactly."""
    if width is None and height is None:
        typer.secho("Provide --width and/or --height.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    im = _load_or_exit(input)
    im.resize(width=width, height=height)
    _save(im, output)


@app.command()
def convert(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    mode: str = typer.Option(..., help="Target color mode: gray, hsv, rgb, lab, bgr."),
):
    """Convert an image's color space."""
    im = _load_or_exit(input)
    try:
        im.convert_color(mode)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _save(im, output)


@app.command()
def rotate(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    angle: float = typer.Argument(..., help="Rotation angle in degrees (counter-clockwise)."),
    scale: float = typer.Option(1.0, help="Scale factor applied during rotation."),
):
    """Rotate an image around its center."""
    im = _load_or_exit(input)
    im.rotate(angle, scale=scale)
    _save(im, output)


@app.command()
def blur(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    method: str = typer.Option("gaussian", help="gaussian, median, box, or bilateral."),
    ksize: int = typer.Option(5, help="Kernel size (rounded up to odd if needed)."),
):
    """Blur an image."""
    im = _load_or_exit(input)
    im.blur(method=method, ksize=ksize)
    _save(im, output)


@app.command()
def edges(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    method: str = typer.Option("canny", help="canny or sobel."),
    t1: int = typer.Option(100, help="First threshold (canny only)."),
    t2: int = typer.Option(200, help="Second threshold (canny only)."),
):
    """Detect edges in an image."""
    im = _load_or_exit(input)
    im.detect_edges(method=method, t1=t1, t2=t2)
    _save(im, output)


@app.command()
def faces(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
    cascade: Optional[Path] = typer.Option(None, help="Path to a custom Haar cascade XML file."),
):
    """Detect faces and save a copy with bounding boxes drawn around them."""
    im = _load_or_exit(input)
    detections = im.detect_faces(cascade_path=str(cascade) if cascade else None)
    for (x, y, w, h) in detections:
        im.draw_rect(int(x), int(y), int(w), int(h))
    typer.echo(f"Found {len(detections)} face(s).")
    _save(im, output)


@app.command()
def ocr(
    input: Path = _INPUT_ARG,
    lang: str = typer.Option("eng", help="Tesseract language code."),
):
    """Extract text from an image. Requires the 'ocr' extra (see README)."""
    im = _load_or_exit(input)
    try:
        typer.echo(im.ocr_text(lang=lang))
    except ImportError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def qr(input: Path = _INPUT_ARG):
    """Decode QR codes / barcodes in an image. Requires the 'qr' extra (see README)."""
    im = _load_or_exit(input)
    try:
        results = im.decode_qr_barcode()
    except ImportError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if not results:
        typer.echo("No QR codes or barcodes found.")
        return
    for r in results:
        typer.echo(f"[{r['type']}] {r['data']}")


@app.command()
def inspect(
    input: Path = _INPUT_ARG,
    output: Path = _OUTPUT_ARG,
):
    """Generate report of an image."""
    im = _load_or_exit(input)
    im.inspect(out_dir=output)

if __name__ == "__main__":
    app()