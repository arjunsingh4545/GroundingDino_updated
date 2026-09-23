import sys
import os
import argparse
from pathlib import Path
import urllib.request

# Ensure the local groundingdino repo is importable
_GDINO_REPO = str(Path(__file__).resolve().parent.parent / "groundingdino")
if _GDINO_REPO not in sys.path:
    sys.path.insert(0, _GDINO_REPO)

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino

import json
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from torchvision.ops import box_convert


# ── Constants ─────────────────────────────────────────────────────────

BANNER = r"""
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ██╗███╗   ███╗ ██████╗ ██████╗                       │
│    ██║████╗ ████║██╔═══██╗██╔══██╗                      │
│    ██║██╔████╔██║██║   ██║██║  ██║                      │
│    ██║██║╚██╔╝██║██║   ██║██║  ██║                      │
│    ██║██║ ╚═╝ ██║╚██████╔╝██████╔╝                      │
│    ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═════╝                       │
│                                                         │
│    Image Modification Tool · Powered by GroundingDINO   │
│    Detect objects. Blur selectively. Revert cleanly.     │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""

# Auto-discover config files from the installed groundingdino package
GDINO_CONFIG_DIR = Path(groundingdino.__file__).parent / "config"

# Default weights dir based on user setup
DEFAULT_WEIGHTS_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "04-06-segment-anything" / "weights"
)

# Model variants ordered by size: largest first for auto-selection.
# On ≥8 GB VRAM, SwinB is preferred; on <8 GB, SwinT is the fallback.
MODEL_VARIANTS = [
    {
        "name": "SwinB",
        "config": str(GDINO_CONFIG_DIR / "GroundingDINO_SwinB_cfg.py"),
        "weights_filename": "groundingdino_swinb_cogcoor.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth",
        "vram": "~8 GB",
    },
    {
        "name": "SwinT",
        "config": str(GDINO_CONFIG_DIR / "GroundingDINO_SwinT_OGC.py"),
        "weights_filename": "groundingdino_swint_ogc.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "vram": "~4 GB",
    },
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Config ────────────────────────────────────────────────────────────


class Config:
    """Centralized pipeline configuration.

    Attributes:
        objects:        Object names to detect and blur.
        custom_prompts: Extra detection targets (e.g., 'watermark', 'logo').
        blur_factor:    0 → none, 1 → opaque (mean-colour fill), 0–1 → proportional.
        blur_type:      'gaussian' | 'pixelate' | 'box'.
        box_threshold:  Min detection confidence to keep a bounding box.
        text_threshold: Min token-similarity to assign a label.
        weights_path:   Path to a .pth file **or** a directory containing weight files.
    """

    objects: list = []
    custom_prompts: list = []
    blur_factor: float = None
    blur_type: str = "gaussian"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    weights_path: str = None
    device: str = "cuda"
    output_dir: str = "output"
    input_path: str = None
    num_workers: int = min(os.cpu_count() or 4, 8)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls()
        cfg.objects = args.objects or []
        cfg.custom_prompts = args.custom_prompts or []
        cfg.blur_factor = args.blur_factor
        cfg.blur_type = args.blur_type
        cfg.box_threshold = args.box_threshold
        cfg.text_threshold = args.text_threshold
        cfg.weights_path = args.weights
        cfg.output_dir = args.output
        cfg.input_path = args.input
        cfg.num_workers = args.workers
        cfg.device = (
            "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        return cfg


# ── PromptGenerator ───────────────────────────────────────────────────


class PromptGenerator:
    """Builds and manages GroundingDINO text prompts.

    GroundingDINO expects detection targets separated by `` . ``
    (e.g. ``"person . car . watermark ."``).

    The prompt merges ``Config.objects`` with any ``custom_prompts``
    (added via CLI or at runtime), deduplicates them, and caches the
    result.  Calling ``add_custom_prompt()`` invalidates the cache so the
    next ``generate_prompt()`` rebuilds it.

    Methods:
        generate_prompt        — forward operation prompt.
        add_custom_prompt      — extend detection targets at runtime.
        generate_revert_prompt — manifest entry for undoing a modification.
    """

    def __init__(self, config: Config):
        self.config = config
        self._prompt: str | None = None
        self._custom_prompts: list[str] = list(config.custom_prompts)

    def generate_prompt(self) -> str:
        """Build the GroundingDINO detection prompt from objects + custom prompts.

        Returns a cached string like ``"person . car . watermark ."``.
        Raises ValueError if both lists are empty.
        """
        if self._prompt is not None:
            return self._prompt

        all_targets = [o.strip().lower() for o in self.config.objects if o.strip()]
        all_targets.extend(o.strip().lower() for o in self._custom_prompts if o.strip())

        # Deduplicate while preserving insertion order
        seen: set[str] = set()
        unique: list[str] = []
        for t in all_targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        if not unique:
            raise ValueError(
                "No detection targets (--objects and --custom-prompts are both empty)"
            )

        self._prompt = " . ".join(unique) + " ."
        return self._prompt

    def add_custom_prompt(self, prompt: str):
        """Extend detection targets at runtime. Invalidates the cached prompt."""
        cleaned = prompt.strip().lower()
        if cleaned:
            self._custom_prompts.append(cleaned)
            self._prompt = None

    def generate_revert_prompt(
        self,
        image_path: str,
        output_path: str,
        boxes: list,
        phrases: list,
    ) -> dict:
        """Create a self-contained revert manifest entry for one image.

        Stores everything needed to trace and undo the modification:
        original path, detection prompt, bounding boxes, and parameters.
        """
        return {
            "original_path": str(Path(image_path).resolve()),
            "output_path": str(Path(output_path).resolve()),
            "prompt": self.generate_prompt(),
            "objects": list(self.config.objects),
            "custom_prompts": list(self._custom_prompts),
            "blur_factor": self.config.blur_factor,
            "blur_type": self.config.blur_type,
            "box_threshold": self.config.box_threshold,
            "text_threshold": self.config.text_threshold,
            "detections": [
                {
                    "phrase": phrase,
                    "box_cxcywh": box if isinstance(box, list) else box.tolist(),
                }
                for box, phrase in zip(boxes, phrases)
            ],
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def load_manifest(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)


# ── ModifyImages ──────────────────────────────────────────────────────


class ModifyImages:
    """Detect objects via GroundingDINO and apply selective blur.

    Features:
        - **Auto model selection** — tries SwinB first (better accuracy);
          on CUDA OOM, falls back to SwinT (fits ≤4 GB VRAM).
        - **Multi-threaded post-processing** — blur + save runs on a
          thread pool, overlapping with the next image's GPU inference.
        - **Manifest-based revert** — originals are never overwritten.
    """

    def __init__(self, config: Config):
        self.config = config
        self.prompt_gen = PromptGenerator(config)
        self.model = None
        self._current_variant: str | None = None
        self._weight_candidates: list[tuple[str, str, str]] = []
        self._candidate_idx: int = 0
        self._model_validated: bool = False
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self._manifest: dict = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "images": {},
        }

    # ── Weight discovery & model loading ─────────────────────────────

    @staticmethod
    def _match_variant(weight_name: str) -> dict | None:
        """Return the MODEL_VARIANTS entry matching a weight filename."""
        name = weight_name.lower()
        for mv in MODEL_VARIANTS:
            # Match on the key differentiator in the filename
            key = mv["weights_filename"].split("_")[1]  # 'swinb' or 'swint'
            if key in name:
                return mv
        return None

    def _download_weight(self, url: str, path: str):
        """Download a weight file with a simple progress print."""
        print(f"  Downloading {Path(path).name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  ✓ Downloaded {Path(path).name}")

    def _discover_weights(self) -> list[tuple[str, str, str]]:
        """Return [(config_path, weights_path, variant_name), ...] ordered biggest→smallest."""
        wp = Path(self.config.weights_path)
        candidates: list[tuple[str, str, str]] = []

        if wp.suffix == ".pth":
            mv = self._match_variant(wp.name) or MODEL_VARIANTS[-1]
            if not wp.exists():
                self._download_weight(mv["url"], str(wp))
            candidates.append((mv["config"], str(wp), mv["name"]))

            # Also check for the other variant in the same directory
            for alt in MODEL_VARIANTS:
                if alt["name"] != mv["name"]:
                    alt_path = wp.parent / alt["weights_filename"]
                    if alt_path.exists():
                        candidates.append((alt["config"], str(alt_path), alt["name"]))

        else:  # Directory
            for mv in MODEL_VARIANTS:
                w = wp / mv["weights_filename"]
                if w.exists():
                    candidates.append((mv["config"], str(w), mv["name"]))

            # Fallback: pick up any .pth file in the directory
            if not candidates:
                for pth in sorted(wp.glob("*.pth")):
                    mv = self._match_variant(pth.name) or MODEL_VARIANTS[-1]
                    candidates.append((mv["config"], str(pth), mv["name"]))

            if not candidates:
                print(f"  No weights found in {wp}. Downloading defaults...")
                for mv in MODEL_VARIANTS:
                    w = wp / mv["weights_filename"]
                    self._download_weight(mv["url"], str(w))
                    candidates.append((mv["config"], str(w), mv["name"]))

        # Ensure SwinB comes before SwinT
        order = {mv["name"]: i for i, mv in enumerate(MODEL_VARIANTS)}
        candidates.sort(key=lambda c: order.get(c[2], 99))
        return candidates

    def _ensure_model(self):
        """Load the best available model variant (largest that fits in VRAM)."""
        if self.model is not None:
            return

        self._weight_candidates = self._discover_weights()
        if not self._weight_candidates:
            raise FileNotFoundError(
                f"No weight files found at: {self.config.weights_path}\n"
                f"  Expected: {' or '.join(mv['weights_filename'] for mv in MODEL_VARIANTS)}"
            )

        # Try each candidate — OOM during load triggers fallback
        for idx, (config_path, weight_path, variant) in enumerate(
            self._weight_candidates
        ):
            vram = next(m["vram"] for m in MODEL_VARIANTS if m["name"] == variant)
            print(f"  Loading {variant} model  (VRAM ≈ {vram}) …")
            try:
                self.model = load_model(
                    config_path, weight_path, device=self.config.device
                )
                self._current_variant = variant
                self._candidate_idx = idx
                # CPU never OOMs, so mark validated immediately
                self._model_validated = self.config.device == "cpu"
                print(f"  {variant} model loaded.\n")
                return
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"  ⚠  OOM loading {variant}, trying smaller variant …")

        raise RuntimeError(
            "All model variants failed to load (CUDA OOM). Try --cpu-only."
        )

    def _predict_with_fallback(self, image_tensor: torch.Tensor, prompt: str):
        """Run prediction. On first call, catches OOM and falls back to a smaller model."""
        kwargs = dict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.config.device,
        )

        if self._model_validated:
            return predict(**kwargs)

        try:
            result = predict(**kwargs)
            self._model_validated = True
            return result
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            torch.cuda.empty_cache()
            del self.model
            self.model = None

            remaining = self._weight_candidates[self._candidate_idx + 1 :]
            if not remaining:
                raise RuntimeError(
                    f"CUDA OOM with {self._current_variant} during inference "
                    f"and no smaller model available. Try --cpu-only."
                ) from e

            config_path, weight_path, variant = remaining[0]
            vram = next(m["vram"] for m in MODEL_VARIANTS if m["name"] == variant)
            print(
                f"\n  ⚠  OOM with {self._current_variant} — falling back to {variant} ({vram}) …"
            )
            self.model = load_model(config_path, weight_path, device=self.config.device)
            self._current_variant = variant
            self._candidate_idx += 1

            kwargs["model"] = self.model
            result = predict(**kwargs)
            self._model_validated = True
            print(f"  ✓  {variant} inference OK.\n")
            return result

    # ── Blur ─────────────────────────────────────────────────────────

    def _apply_blur_inplace(self, image: np.ndarray, box_xyxy):
        """Apply blur to a bounding-box region **in-place**."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(c) for c in box_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        factor = self.config.blur_factor
        if factor <= 0:
            return

        region = image[y1:y2, x1:x2]
        rh, rw = region.shape[:2]

        if factor >= 1.0:
            image[y1:y2, x1:x2] = region.mean(axis=(0, 1)).astype(np.uint8)
            return

        if self.config.blur_type == "gaussian":
            ksize = max(3, int(max(rh, rw) * factor) | 1)
            image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
        elif self.config.blur_type == "pixelate":
            pixel_size = max(2, int(min(rh, rw) * factor))
            small = cv2.resize(
                region,
                (max(1, rw // pixel_size), max(1, rh // pixel_size)),
                interpolation=cv2.INTER_LINEAR,
            )
            image[y1:y2, x1:x2] = cv2.resize(
                small, (rw, rh), interpolation=cv2.INTER_NEAREST
            )
        elif self.config.blur_type == "box":
            ksize = max(3, int(max(rh, rw) * factor) | 1)
            image[y1:y2, x1:x2] = cv2.blur(region, (ksize, ksize))

    # ── Thread-pool task ─────────────────────────────────────────────

    def _blur_and_save(self, image_source, xyxy, phrases, logits_np, out_path):
        """Apply blur to every detection and write to disk. Runs in a worker thread.

        Returns a list of (phrase, confidence) tuples for logging.
        """
        result = image_source.copy()
        log = []
        for box, phrase, logit in zip(xyxy, phrases, logits_np):
            self._apply_blur_inplace(result, box)
            log.append((phrase, float(logit)))
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return log

    # ── Core processing ──────────────────────────────────────────────

    def modify_image(self, image_path: str) -> str:
        """Process a single image (no threading). Returns the output path."""
        self._ensure_model()
        prompt = self.prompt_gen.generate_prompt()

        image_source, image_tensor = load_image(image_path)
        boxes, logits, phrases = self._predict_with_fallback(image_tensor, prompt)

        if len(boxes) == 0:
            print("    ⚠  No objects detected")

        h, w = image_source.shape[:2]
        pixel_boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(image_path).name

        result = image_source.copy()
        for box, phrase, logit in zip(xyxy, phrases, logits):
            self._apply_blur_inplace(result, box)
            print(f"    ✓  Blurred [{phrase}]  (confidence: {logit:.3f})")

        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        self._manifest["images"][str(out_path.resolve())] = (
            self.prompt_gen.generate_revert_prompt(
                image_path,
                str(out_path),
                boxes.tolist(),
                phrases,
            )
        )
        return str(out_path)

    def modify_images(self, input_path: str) -> list[str]:
        """Process a file or directory.

        For directories: GPU inference runs sequentially while blur+save
        runs on the thread pool, overlapping CPU and GPU work.
        """
        path = Path(input_path)
        results: list[str] = []

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported format: {path.suffix}")
            print(f"\n  Image: {path.name}")
            results.append(self.modify_image(str(path)))

        elif path.is_dir():
            images = sorted(
                f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not images:
                raise FileNotFoundError(f"No supported images in: {path}")

            total = len(images)
            print(f"\n  Found {total} image(s) in: {path}\n")

            self._ensure_model()
            prompt = self.prompt_gen.generate_prompt()

            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Phase 1: Infer on GPU (sequential) → submit blur+save to thread pool.
            # While the pool processes image N, the main thread loads+infers image N+1.
            pending: list[tuple] = []

            for idx, img_path in enumerate(images, 1):
                print(f"  [{idx}/{total}] {img_path.name}")

                image_source, image_tensor = load_image(str(img_path))
                boxes, logits, phrases = self._predict_with_fallback(
                    image_tensor, prompt
                )

                n = len(boxes)
                if n == 0:
                    print("    ⚠  No objects detected")
                else:
                    for phrase, logit in zip(phrases, logits):
                        print(f"    •  Detected [{phrase}]  ({logit:.3f})")

                h, w = image_source.shape[:2]
                pixel_boxes = boxes * torch.Tensor([w, h, w, h])
                xyxy = box_convert(pixel_boxes, "cxcywh", "xyxy").numpy()

                out_path = str(out_dir / img_path.name)
                future = self._executor.submit(
                    self._blur_and_save,
                    image_source,
                    xyxy,
                    phrases,
                    logits.numpy(),
                    out_path,
                )
                pending.append(
                    (future, str(img_path), out_path, boxes.tolist(), phrases)
                )

            # Phase 2: Collect thread pool results and build manifest.
            print(f"\n  Waiting for blur + save threads …")
            for future, img_path, out_path, box_list, phrases in pending:
                future.result()  # raises on thread error
                results.append(out_path)
                self._manifest["images"][str(Path(out_path).resolve())] = (
                    self.prompt_gen.generate_revert_prompt(
                        img_path,
                        out_path,
                        box_list,
                        phrases,
                    )
                )

        else:
            raise FileNotFoundError(f"Path not found: {path}")

        # Persist the manifest
        manifest_path = Path(self.config.output_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        print(f"\n  {'─' * 48}")
        print(f"  ✓  {len(results)} image(s) processed")
        print(f"  ✓  Output dir  → {Path(self.config.output_dir).resolve()}")
        print(f"  ✓  Manifest    → {manifest_path.resolve()}")
        return results

    @staticmethod
    def revert_changes(manifest_path: str, output_dir: str | None = None):
        """Restore original images using a saved manifest.

        Since originals are never overwritten, this copies them into
        *output_dir* (defaults to ``<manifest_dir>/reverted/``).
        """
        manifest = PromptGenerator.load_manifest(manifest_path)

        if output_dir is None:
            output_dir = str(Path(manifest_path).parent / "reverted")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        reverted = 0
        for _, entry in manifest.get("images", {}).items():
            original = Path(entry["original_path"])
            if not original.exists():
                print(f"    ✗  Original missing: {original}")
                continue

            dest = out / original.name
            img = cv2.imread(str(original))
            cv2.imwrite(str(dest), img)
            reverted += 1
            print(f"    ✓  Restored: {original.name}")

        print(f"\n  {'─' * 48}")
        print(f"  Reverted {reverted} image(s) → {out.resolve()}")

    def __del__(self):
        self._executor.shutdown(wait=False)


# ── CLI ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imod",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Blur all people and cars in a single image
  python src/main.py -i photo.jpg -w weights/ --objects person car --blur-factor 0.6

  # Pixelate faces, also detect watermarks
  python src/main.py -i imgs/ -w weights/groundingdino_swint_ogc.pth \\
      --objects face --custom-prompts watermark logo \\
      --blur-factor 0.8 --blur-type pixelate

  # Revert a previous run using its manifest
  python src/main.py --revert output/manifest.json

  # Force CPU and use 4 worker threads
  python src/main.py -i img.png -w weights/ --objects dog cat \\
      --blur-factor 0.4 --blur-type box --cpu-only --workers 4
        """,
    )

    # ── Mode ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--revert",
        type=str,
        metavar="MANIFEST",
        default=None,
        help="Revert a previous run using the specified manifest.json",
    )
    parser.add_argument(
        "--revert-output",
        type=str,
        metavar="DIR",
        default=None,
        help="Directory for reverted images  (default: <manifest_dir>/reverted/)",
    )

    # ── Input / Output ───────────────────────────────────────────────
    io_grp = parser.add_argument_group("input / output")
    io_grp.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to a single image or a directory of images",
    )
    io_grp.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory for modified images  (default: output/)",
    )

    # ── Detection targets ────────────────────────────────────────────
    det_grp = parser.add_argument_group("detection targets")
    det_grp.add_argument(
        "--objects",
        nargs="+",
        type=str,
        help="Primary object names to detect and blur  (space-separated)",
    )
    det_grp.add_argument(
        "--custom-prompts",
        nargs="+",
        type=str,
        default=[],
        help="Extra detection targets  (e.g. watermark logo text)",
    )

    # ── Blur parameters ──────────────────────────────────────────────
    blur_grp = parser.add_argument_group("blur parameters")
    blur_grp.add_argument(
        "--blur-factor",
        type=float,
        default=0.5,
        metavar="F",
        help="Blur intensity: 0 = none, 1 = opaque  (default: 0.5)",
    )
    blur_grp.add_argument(
        "--blur-type",
        choices=["gaussian", "pixelate", "box"],
        default="gaussian",
        help="Blur algorithm to apply  (default: gaussian)",
    )

    # ── Detection thresholds ─────────────────────────────────────────
    thr_grp = parser.add_argument_group("detection thresholds")
    thr_grp.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        metavar="T",
        help="Min detection confidence to keep a box  (default: 0.35)",
    )
    thr_grp.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        metavar="T",
        help="Min token-similarity to assign a label  (default: 0.25)",
    )

    # ── Model ────────────────────────────────────────────────────────
    mdl_grp = parser.add_argument_group("model configuration")
    mdl_grp.add_argument(
        "-w",
        "--weights",
        type=str,
        metavar="PATH",
        default=DEFAULT_WEIGHTS_DIR,
        help="Path to a .pth weights file or a directory containing them. "
        "If a directory, tries SwinB first (≥8 GB VRAM), falls back to SwinT.",
    )
    mdl_grp.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU inference (skips GPU entirely)",
    )
    mdl_grp.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        metavar="N",
        help=f"Thread-pool workers for blur + save  (default: {min(os.cpu_count() or 4, 8)})",
    )

    return parser


def validate_args(args: argparse.Namespace):
    if args.revert:
        if not Path(args.revert).exists():
            sys.exit(f"  ✗  Manifest not found: {args.revert}")
        return

    if not args.input:
        sys.exit("  ✗  --input / -i is required.  Run with --help for usage.")
    if not args.objects and not args.custom_prompts:
        sys.exit("  ✗  At least one of --objects or --custom-prompts is required.")
    if not 0 <= args.blur_factor <= 1:
        sys.exit("  ✗  --blur-factor must be between 0 and 1.")
    if not Path(args.input).exists():
        sys.exit(f"  ✗  Input not found: {args.input}")

    wp = Path(args.weights)
    if not wp.exists():
        if wp.suffix == ".pth":
            wp.parent.mkdir(parents=True, exist_ok=True)
        else:
            wp.mkdir(parents=True, exist_ok=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(BANNER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    validate_args(args)

    # ── Revert mode ──────────────────────────────────────────────────
    if args.revert:
        print("  Mode     │  REVERT")
        print(f"  Manifest │  {args.revert}")
        print()
        ModifyImages.revert_changes(args.revert, args.revert_output)
        return

    # ── Blur mode ────────────────────────────────────────────────────
    config = Config.from_args(args)

    all_targets = (args.objects or []) + (args.custom_prompts or [])
    device_label = (
        "CPU (forced)"
        if args.cpu_only
        else ("CUDA" if torch.cuda.is_available() else "CPU (no GPU)")
    )

    print(f"  Mode     │  BLUR")
    print(f"  Targets  │  {', '.join(all_targets)}")
    print(f"  Blur     │  {args.blur_type}  @  {args.blur_factor:.0%}")
    print(f"  Device   │  {device_label}")
    print(f"  Workers  │  {config.num_workers}")
    print(f"  Output   │  {Path(args.output).resolve()}")
    print()

    modifier = ModifyImages(config)
    modifier.modify_images(config.input_path)


if __name__ == "__main__":
    main()
