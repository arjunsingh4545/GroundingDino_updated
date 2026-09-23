#!/usr/bin/env python3
"""
Download GroundingDINO checkpoint weights into a local model_weights/ directory.

Fixes the two failure modes hit so far:
  - Wrong release tag: SwinB was published under v0.1.0-alpha2, not
    v0.1.0-alpha (which only has SwinT) -- that mismatch is what caused
    the 404.
  - Corrupted/partial files: downloads go to a .part file and are
    verified as valid zip archives (what torch.load actually needs)
    before being renamed into place, so an interrupted download never
    leaves a broken .pth sitting at the real path.

Usage:
    python download_weights.py                  # both variants -> ./model_weights
    python download_weights.py --variant swinb   # just SwinB
    python download_weights.py -o /custom/path
"""

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

WEIGHTS = {
    "swint": {
        "filename": "groundingdino_swint_ogc.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    },
    "swinb": {
        "filename": "groundingdino_swinb_cogcoor.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
    },
}


def download_one(name: str, url: str, dest: Path) -> None:
    if dest.exists():
        print(
            f"  already present, skipping: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)"
        )
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {name} -> {dest}")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        print(
            f"\r    {done / total_size * 100:5.1f}%  ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"\n  ✗ {name}: HTTP {e.code} for {url}\n"
            f"    Release asset may have moved -- check "
            f"https://github.com/IDEA-Research/GroundingDINO/releases for the current URL."
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # This is the exact check that would've caught the earlier corrupted-checkpoint
    # error ("failed finding central directory") before it ever reached torch.load.
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"\n  ✗ {name}: downloaded file isn't a valid archive "
            f"(interrupted download or an HTML error page saved as .pth). Re-run to retry."
        )

    tmp.replace(dest)  # atomic -- only becomes the real file once verified
    print(f"  ✓ {name} verified and saved ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download GroundingDINO checkpoint weights."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="model_weights",
        help="Output directory (default: ./model_weights)",
    )
    parser.add_argument("--variant", choices=["swint", "swinb", "both"], default="both")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in WEIGHTS if args.variant == "both" else [args.variant]:
        info = WEIGHTS[v]
        download_one(v, info["url"], out_dir / info["filename"])

    print(
        f"\nDone. Point your script at this directory:\n"
        f"  python -m src.main -i <image> -w {out_dir.resolve()} --objects ..."
    )


if __name__ == "__main__":
    main()
