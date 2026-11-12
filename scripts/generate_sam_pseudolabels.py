from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
SPLIT_MANIFEST = Path("reports/phase3_split/split_manifest.csv")
DEFAULT_CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
DEFAULT_OUTPUT = Path("reports/phase3_pseudolabels/pilot")
THRESHOLD_C = 80.0
MINIMUM_COMPONENT_AREA = 50
IMAGE_SIZE = (640, 512)
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
METADATA_FIELDS = [
    "label", "stem", "split", "timestamp", "threshold_c", "minimum_component_area",
    "rgb_source", "thermal_source", "mask_path", "status", "sam_run", "failure",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "sam_predicted_iou",
    "inference_seconds", "sam_mask_pixels", "sam_mask_area_percent",
    "thermal_candidate_pixels", "thermal_candidate_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM ViT-B pseudo-labels from thermal boxes.")
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ordered_rows(path: Path, limit_per_split: int | None) -> list[dict[str, str]]:
    rows = read_rows(path)
    if len(rows) != 738:
        raise ValueError(f"Frozen split manifest must contain 738 rows, found {len(rows)}")
    if len({(row["label"], row["stem"]) for row in rows}) != 738:
        raise ValueError("Frozen split manifest contains duplicate quartet keys")
    rows.sort(key=lambda row: (SPLIT_ORDER[row["split"]], row["timestamp"], row["label"], row["stem"]))
    selected = []
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        selected.extend(split_rows[:limit_per_split] if limit_per_split else split_rows)
    return selected


def output_mask_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / row["split"] / f"{row['label'].lower().replace(' ', '_')}_{row['stem']}.png"


def is_completed(row: dict[str, str]) -> bool:
    if row.get("status") == "success":
        return row.get("sam_run") == "True" and Path(row.get("mask_path", "")).is_file()
    if row.get("status") == "no_valid_box_background":
        return Path(row.get("mask_path", "")).is_file()
    return row.get("status") == "no_valid_box" and row.get("label") == "Fire" and not Path(row.get("mask_path", "")).is_file()


def load_input_paths(row: dict[str, str]) -> tuple[Path, Path]:
    label = row["label"]
    stem = row["stem"]
    return (
        DATASET_ROOT / label / "RGB" / "Corrected FOV" / f"{stem}.JPG",
        DATASET_ROOT / label / "Thermal" / "Celsius TIFF" / f"{stem}.TIFF",
    )


def largest_component_box(thermal: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    candidate_mask = thermal >= THRESHOLD_C
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return candidate_mask, None
    component_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x = int(stats[component_index, cv2.CC_STAT_LEFT])
    y = int(stats[component_index, cv2.CC_STAT_TOP])
    width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    if area < MINIMUM_COMPONENT_AREA:
        return candidate_mask, None
    return candidate_mask, np.asarray([x, y, x + width - 1, y + height - 1], dtype=np.float32)


def write_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise IOError(f"Failed to write mask: {path}")


def process_sample(
    predictor: SamPredictor | None,
    row: dict[str, str],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    rgb_path, thermal_path = load_input_paths(row)
    sample_dir = output_dir / row["split"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask_path = sample_dir / f"{row['label'].lower().replace(' ', '_')}_{row['stem']}.png"
    thermal_mask_path = sample_dir / f"{row['label'].lower().replace(' ', '_')}_{row['stem']}_thermal_candidate.png"
    result: dict[str, Any] = {
        "label": row["label"],
        "stem": row["stem"],
        "split": row["split"],
        "timestamp": row["timestamp"],
        "threshold_c": THRESHOLD_C,
        "minimum_component_area": MINIMUM_COMPONENT_AREA,
        "rgb_source": str(rgb_path),
        "thermal_source": str(thermal_path),
        "mask_path": str(mask_path),
        "status": "failed",
        "sam_run": False,
        "failure": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "sam_predicted_iou": "",
        "inference_seconds": "",
        "sam_mask_pixels": "",
        "sam_mask_area_percent": "",
    }
    try:
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None or thermal is None:
            raise RuntimeError("Failed to read corrected RGB or Celsius TIFF")
        if image_bgr.shape[:2] != thermal.shape[:2] or image_bgr.shape[:2] != (512, 640):
            raise RuntimeError(f"RGB/TIFF alignment or dimensions invalid: {image_bgr.shape}, {thermal.shape}")
        candidate_mask, bbox = largest_component_box(thermal.astype(np.float32, copy=False))
        write_mask(thermal_mask_path, candidate_mask)
        result["thermal_candidate_pixels"] = int(candidate_mask.sum())
        result["thermal_candidate_percent"] = float(candidate_mask.mean() * 100)
        if bbox is None:
            if row["label"] == "No Fire":
                write_mask(mask_path, np.zeros((512, 640), dtype=bool))
                result["status"] = "no_valid_box_background"
            else:
                if mask_path.is_file():
                    mask_path.unlink()
                result["status"] = "no_valid_box"
            return result

        result.update({
            "bbox_x1": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x2": int(bbox[2]),
            "bbox_y2": int(bbox[3]),
        })
        if predictor is None:
            raise RuntimeError("SAM predictor is unavailable")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictor.set_image(image)
        masks, scores, _logits = predictor.predict(box=bbox, multimask_output=False)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        sam_mask = masks[0].astype(bool)
        write_mask(mask_path, sam_mask)
        result.update({
            "status": "success",
            "sam_run": True,
            "sam_predicted_iou": float(scores[0]),
            "inference_seconds": inference_seconds,
            "sam_mask_pixels": int(sam_mask.sum()),
            "sam_mask_area_percent": float(sam_mask.mean() * 100),
        })
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    args = parse_args()
    if args.limit_per_split is not None and args.limit_per_split < 1:
        raise ValueError("--limit-per-split must be positive")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    selected = ordered_rows(args.manifest, args.limit_per_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.csv"
    existing = {}
    if args.resume and metadata_path.is_file():
        existing = {
            (row["label"], row["stem"], row["split"]): row
            for row in read_rows(metadata_path)
        }
    results = []
    pending = []
    for row in selected:
        key = (row["label"], row["stem"], row["split"])
        prior = existing.get(key)
        if prior is not None and is_completed(prior):
            results.append(prior)
        else:
            pending.append(row)
    model = None
    predictor = None
    try:
        if pending:
            model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device)
            model.eval()
            predictor = SamPredictor(model)
        for row in pending:
            results.append(process_sample(predictor, row, args.output_dir, args.device))
    finally:
        if predictor is not None:
            del predictor
        if model is not None:
            del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    results.sort(key=lambda row: (SPLIT_ORDER[row["split"]], row["timestamp"], row["label"], row["stem"]))
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    split_counts = {}
    for split in ("train", "validation", "test"):
        split_results = [row for row in results if row["split"] == split]
        split_counts[split] = {
            "requested": len(split_results),
            "success": sum(row["status"] == "success" for row in split_results),
            "no_valid_box": sum(row["status"] == "no_valid_box" for row in split_results),
            "no_valid_box_background": sum(row["status"] == "no_valid_box_background" for row in split_results),
            "failed": sum(row["status"] == "failed" for row in split_results),
        }
    summary = {
        "source_manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "model": "vit_b",
        "device": args.device, "threshold_c": THRESHOLD_C, "minimum_component_area": MINIMUM_COMPONENT_AREA,
        "limit_per_split": args.limit_per_split, "resumed_completed": len(results) - len(pending),
        "processed_this_run": len(pending), "total": len(results), "split_counts": split_counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary_by_split.json").write_text(json.dumps(split_counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()