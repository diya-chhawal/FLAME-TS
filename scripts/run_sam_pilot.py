from __future__ import annotations

import csv
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
MANIFEST_PATH = Path("reports/phase1/quartet_manifest.csv")
COMPONENT_METRICS_PATH = Path("reports/phase2_5/component_filter_metrics.csv")
OUTPUT_DIR = Path("reports/phase3_sam_pilot")
CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
PILOT_CONFIGURATIONS = (
    (60.0, 100),
    (80.0, 50),
    (100.0, 10),
)
FIRE_CATEGORIES = ("easy", "weak", "fragmented", "multi_region", "border_touching")
NO_FIRE_COUNT = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_cases(
    manifest: list[dict[str, str]], metrics: list[dict[str, str]], threshold: float, minimum_area: int
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    fire = [
        row for row in metrics
        if row["label"] == "Fire"
        and float(row["threshold_c"]) == threshold
        and int(row["minimum_component_area"]) == minimum_area
    ]
    by_stem = {row["stem"]: row for row in manifest if row["label"] == "Fire"}
    used: set[str] = set()

    def add(category: str, candidates: list[dict[str, str]]) -> None:
        candidates = [row for row in candidates if row["stem"] not in used]
        if candidates:
            metric_row = sorted(candidates, key=lambda row: row["stem"])[0]
            chosen = dict(by_stem[metric_row["stem"]])
            chosen["case_category"] = category
            selected.append(chosen)
            used.add(metric_row["stem"])

    valid = [row for row in fire if row["has_retained_component"].lower() == "true"]
    add("easy", sorted(valid, key=lambda row: (-int(row["largest_retained_component_area_pixels"]), row["stem"])))
    add("weak", sorted(valid, key=lambda row: (int(row["largest_retained_component_area_pixels"]), row["stem"])))
    add("fragmented", sorted(valid, key=lambda row: (-int(row["retained_component_count"]), row["stem"])))
    multi = [row for row in valid if int(row["retained_component_count"]) >= 2]
    add("multi_region", sorted(multi, key=lambda row: (-int(row["retained_component_count"]), row["stem"])))
    border = [
        row for row in valid
        if int(row["largest_retained_bbox_x_min"]) == 0
        or int(row["largest_retained_bbox_y_min"]) == 0
        or int(row["largest_retained_bbox_x_max"]) == 639
        or int(row["largest_retained_bbox_y_max"]) == 511
    ]
    add("border_touching", sorted(border, key=lambda row: row["stem"]))

    no_fire_manifest = sorted((row for row in manifest if row["label"] == "No Fire"), key=lambda row: row["stem"])
    for row in no_fire_manifest[:NO_FIRE_COUNT]:
        selected_row = dict(row)
        selected_row["case_category"] = "no_fire"
        selected.append(selected_row)
    return selected


def box_from_metric(row: dict[str, str]) -> np.ndarray | None:
    if row["has_retained_component"].lower() != "true":
        return None
    return np.array(
        [
            int(row["largest_retained_bbox_x_min"]),
            int(row["largest_retained_bbox_y_min"]),
            int(row["largest_retained_bbox_x_max"]),
            int(row["largest_retained_bbox_y_max"]),
        ],
        dtype=np.float32,
    )


def save_gray(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), (image.astype(np.uint8) * 255))


def draw_overlay(image: np.ndarray, box: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    overlay = image.copy()
    if mask is not None:
        color = np.zeros_like(overlay)
        color[:, :, 1] = 255
        overlay = np.where(mask[:, :, None], (0.45 * color + 0.55 * overlay).astype(np.uint8), overlay)
    x0, y0, x1, y1 = [int(value) for value in box]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
    return overlay


def process_case(
    predictor: SamPredictor,
    case: dict[str, str],
    metric: dict[str, str],
    threshold: float,
    minimum_area: int,
    device: str,
) -> dict[str, Any]:
    case_dir = OUTPUT_DIR / f"t{int(threshold)}_a{minimum_area}" / f"{case['label'].lower().replace(' ', '_')}_{case['stem']}_{case['case_category']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = DATASET_ROOT / case["corrected_rgb"]
    thermal_path = DATASET_ROOT / case["thermal_tiff"]
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    thermal = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
    result: dict[str, Any] = {
        "label": case["label"],
        "stem": case["stem"],
        "case_category": case["case_category"],
        "threshold_c": threshold,
        "minimum_component_area": minimum_area,
        "rgb_source": case["corrected_rgb"],
        "thermal_source": case["thermal_tiff"],
        "device": device,
        "checkpoint": str(CHECKPOINT),
        "rgb_shape": list(image.shape) if image is not None else None,
        "thermal_shape": list(thermal.shape) if thermal is not None else None,
        "alignment_verified": bool(image is not None and thermal is not None and image.shape[:2] == thermal.shape[:2] == (512, 640)),
        "failures": [],
        "warnings": [],
    }
    shutil.copy2(rgb_path, case_dir / "rgb_corrected_fov.JPG")
    if image is None or thermal is None:
        result["failures"].append("Could not read RGB or thermal input")
        return result
    mask = thermal >= threshold
    save_gray(case_dir / "thermal_threshold_mask.png", mask)
    result["thermal_foreground_pixels"] = int(mask.sum())
    result["thermal_foreground_percent"] = float(mask.mean() * 100)
    box = box_from_metric(metric)
    if box is None:
        result["failures"].append("No valid largest-component box; SAM not run")
        cv2.imwrite(str(case_dir / "thermal_box.png"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return result
    result["box_xyxy"] = [int(value) for value in box]
    result["box_width"] = int(box[2] - box[0] + 1)
    result["box_height"] = int(box[3] - box[1] + 1)
    result["box_area_percent"] = float(result["box_width"] * result["box_height"] / (640 * 512) * 100)
    box_image = image.copy()
    x0, y0, x1, y1 = [int(value) for value in box]
    cv2.rectangle(box_image, (x0, y0), (x1, y1), (255, 255, 0), 2)
    cv2.imwrite(str(case_dir / "thermal_box.png"), box_image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictor.set_image(rgb)
        if device == "cuda":
            torch.cuda.synchronize()
        masks, scores, _logits = predictor.predict(box=box, multimask_output=False)
        if device == "cuda":
            torch.cuda.synchronize()
        result["inference_seconds"] = time.perf_counter() - start
        result["returned_mask_count"] = int(len(masks))
        result["confidence_scores"] = [float(score) for score in scores]
        sam_mask = masks[0].astype(bool)
        valid = bool(sam_mask.shape == (512, 640) and np.isfinite(sam_mask).all() and sam_mask.any())
        result["mask_valid"] = valid
        result["sam_mask_pixels"] = int(sam_mask.sum())
        result["sam_mask_area_percent"] = float(sam_mask.mean() * 100)
        if not valid:
            result["warnings"].append("SAM returned an empty or invalid mask")
        save_gray(case_dir / "sam_mask.png", sam_mask)
        overlay = draw_overlay(image, box, sam_mask)
        cv2.imwrite(str(case_dir / "rgb_box_sam_overlay.png"), overlay)
        return result
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        return result


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; SAM pilot requires the RTX 3050")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(MANIFEST_PATH)
    metrics = read_csv(COMPONENT_METRICS_PATH)
    device = "cuda"
    model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to(device)
    model.eval()
    predictor = SamPredictor(model)
    all_results: list[dict[str, Any]] = []
    try:
        for threshold, minimum_area in PILOT_CONFIGURATIONS:
            metric_by_stem = {
                row["stem"]: row for row in metrics
                if row["label"] == "Fire"
                and float(row["threshold_c"]) == threshold
                and int(row["minimum_component_area"]) == minimum_area
            }
            cases = select_cases(manifest, metrics, threshold, minimum_area)
            for case in cases:
                metric = metric_by_stem.get(case["stem"])
                if case["label"] == "No Fire":
                    metric = {
                        "has_retained_component": "False",
                    }
                if metric is None:
                    continue
                all_results.append(process_case(predictor, case, metric, threshold, minimum_area, device))
    finally:
        del predictor
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with (OUTPUT_DIR / "pilot_results.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "checkpoint": str(CHECKPOINT),
                "model": "vit_b",
                "device": device,
                "configurations": [
                    {"threshold_c": threshold, "minimum_component_area": minimum_area}
                    for threshold, minimum_area in PILOT_CONFIGURATIONS
                ],
                "results": all_results,
            },
            stream,
            indent=2,
        )
    with (OUTPUT_DIR / "pilot_results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for result in all_results for key in result})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(json.dumps({"cases": len(all_results), "successes": sum(not row["failures"] for row in all_results), "failures": sum(bool(row["failures"]) for row in all_results), "output": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()