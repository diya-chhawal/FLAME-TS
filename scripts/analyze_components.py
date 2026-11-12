from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
MANIFEST_PATH = Path("reports/phase1/quartet_manifest.csv")
CONFIG_PATH = Path("configs/phase2_5_component_analysis.json")
OUTPUT_DIR = Path("reports/phase2_5")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = tuple(float(value) for value in config["candidate_thresholds_c"])
    areas = tuple(int(value) for value in config["minimum_component_areas"])
    connectivity = int(config["connectivity"])
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not thresholds or not areas or any(area < 1 for area in areas):
        raise ValueError("thresholds and minimum areas must be nonempty and positive")
    return {
        "candidate_thresholds_c": thresholds,
        "minimum_component_areas": areas,
        "connectivity": connectivity,
        "visualization_cases_per_category": int(config["visualization_cases_per_category"]),
    }


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["quartet_status"] != "complete" for row in rows):
        raise ValueError("Phase 1 manifest contains incomplete quartets")
    return rows


def component_data(values: np.ndarray, threshold: float, connectivity: int) -> tuple[np.ndarray, list[int]]:
    structure = (
        np.ones((3, 3), dtype=np.uint8)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    labels, count = ndimage.label(values >= threshold, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    return labels, [int(size) for size in sizes]


def metrics_for_filter(
    label: str,
    stem: str,
    threshold: float,
    minimum_area: int,
    labels: np.ndarray,
    sizes: list[int],
) -> dict[str, Any]:
    retained_labels = [index + 1 for index, size in enumerate(sizes) if size >= minimum_area]
    retained = np.isin(labels, retained_labels)
    retained_sizes = [sizes[index - 1] for index in retained_labels]
    retained_pixels = int(np.count_nonzero(retained))
    image_area = labels.size
    if retained_labels:
        largest_index = int(np.argmax(retained_sizes))
        largest_area = retained_sizes[largest_index]
        coordinates = np.argwhere(labels == retained_labels[largest_index])
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        bbox_width = int(x_max - x_min + 1)
        bbox_height = int(y_max - y_min + 1)
        bbox_area = bbox_width * bbox_height
        bbox_values = (int(x_min), int(y_min), int(x_max), int(y_max))
    else:
        largest_area = bbox_area = bbox_width = bbox_height = 0
        bbox_values = (0, 0, 0, 0)
    return {
        "label": label,
        "stem": stem,
        "threshold_c": threshold,
        "minimum_component_area": minimum_area,
        "has_retained_component": bool(retained_labels),
        "retained_component_count": len(retained_labels),
        "largest_retained_component_area_pixels": largest_area,
        "retained_foreground_pixels": retained_pixels,
        "retained_foreground_percent": retained_pixels * 100 / image_area,
        "largest_retained_bbox_area_pixels": bbox_area,
        "largest_retained_bbox_area_percent": bbox_area * 100 / image_area,
        "largest_retained_bbox_x_min": bbox_values[0],
        "largest_retained_bbox_y_min": bbox_values[1],
        "largest_retained_bbox_x_max": bbox_values[2],
        "largest_retained_bbox_y_max": bbox_values[3],
        "largest_retained_bbox_width": bbox_width,
        "largest_retained_bbox_height": bbox_height,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], thresholds: tuple[float, ...], areas: tuple[int, ...]) -> list[dict[str, Any]]:
    summary = []
    for threshold in thresholds:
        for minimum_area in areas:
            for label in ("Fire", "No Fire"):
                selected = [
                    row
                    for row in rows
                    if row["threshold_c"] == threshold
                    and row["minimum_component_area"] == minimum_area
                    and row["label"] == label
                ]
                summary.append(
                    {
                        "label": label,
                        "threshold_c": threshold,
                        "minimum_component_area": minimum_area,
                        "samples_with_retained_component": sum(row["has_retained_component"] for row in selected),
                        "samples_total": len(selected),
                        "mean_retained_components": float(np.mean([row["retained_component_count"] for row in selected])),
                        "median_retained_components": float(np.median([row["retained_component_count"] for row in selected])),
                        "mean_largest_retained_area_pixels": float(np.mean([row["largest_retained_component_area_pixels"] for row in selected])),
                        "median_largest_retained_area_pixels": float(np.median([row["largest_retained_component_area_pixels"] for row in selected])),
                        "mean_retained_foreground_percent": float(np.mean([row["retained_foreground_percent"] for row in selected])),
                        "median_retained_foreground_percent": float(np.median([row["retained_foreground_percent"] for row in selected])),
                        "mean_bbox_area_percent": float(np.mean([row["largest_retained_bbox_area_percent"] for row in selected])),
                        "median_bbox_area_percent": float(np.median([row["largest_retained_bbox_area_percent"] for row in selected])),
                    }
                )
    return summary


def render_case(
    row: dict[str, str],
    values: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, Any],
    output_name: str,
) -> None:
    with Image.open(DATASET_ROOT / row["corrected_rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    threshold = float(metrics["threshold_c"])
    minimum_area = int(metrics["minimum_component_area"])
    before = values >= threshold
    retained_labels = [
        index + 1
        for index, size in enumerate(np.bincount(labels.ravel())[1:])
        if int(size) >= minimum_area
    ]
    after = np.isin(labels, retained_labels)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    figure.suptitle(
        f"{row['label']} {row['stem']} | threshold={threshold:g} C | minimum area={minimum_area} px"
    )
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Corrected-FOV RGB")
    axes[0, 1].imshow(values, cmap="inferno")
    axes[0, 1].set_title("Radiometric temperature")
    axes[0, 2].imshow(before, cmap="gray")
    axes[0, 2].set_title("Before filtering")
    axes[1, 0].imshow(after, cmap="gray")
    axes[1, 0].set_title("After component filtering")
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(np.ma.masked_where(~after, after), cmap="spring", alpha=0.55)
    axes[1, 1].set_title("Retained region over RGB")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0,
        1,
        "\n".join(
            [
                f"components: {metrics['retained_component_count']}",
                f"largest area: {metrics['largest_retained_component_area_pixels']} px",
                f"foreground: {metrics['retained_foreground_percent']:.4f}%",
                f"bbox area: {metrics['largest_retained_bbox_area_percent']:.4f}%",
            ]
        ),
        va="top",
        family="monospace",
    )
    for axis in axes.flat:
        if axis.images:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.savefig(OUTPUT_DIR / "visualizations" / f"{output_name}.png", dpi=140)
    plt.close(figure)


def select_cases(
    rows: list[dict[str, str]],
    metrics: list[dict[str, Any]],
    labels_by_key: dict[tuple[str, str, float], np.ndarray],
    config: dict[str, Any],
) -> list[tuple[dict[str, str], dict[str, Any], np.ndarray, str]]:
    row_by_key = {(row["label"], row["stem"]): row for row in rows}
    selected = []
    for threshold in config["candidate_thresholds_c"]:
        for minimum_area in config["minimum_component_areas"]:
            candidates = [
                item for item in metrics
                if item["threshold_c"] == threshold and item["minimum_component_area"] == minimum_area
            ]
            for label, category, sort_key, reverse in (
                ("Fire", "fire_fragmented", "retained_component_count", True),
                ("No Fire", "no_fire_largest", "retained_foreground_percent", True),
            ):
                label_candidates = [item for item in candidates if item["label"] == label]
                if label_candidates:
                    item = sorted(label_candidates, key=lambda value: value[sort_key], reverse=reverse)[0]
                    key = (label, item["stem"])
                    selected.append((row_by_key[key], item, labels_by_key[(label, item["stem"], threshold)], f"{category}_{int(threshold)}c_{minimum_area}px_{item['stem']}"))
    unique = {name: (row, item, labels, name) for row, item, labels, name in selected}
    return list(unique.values())


def main() -> None:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)
    rows = load_manifest()
    metrics = []
    labels_by_key = {}
    values_by_key = {}
    for row in rows:
        key = (row["label"], row["stem"])
        with Image.open(DATASET_ROOT / row["thermal_tiff"]) as image:
            values = np.asarray(image, dtype=np.float32)
        values_by_key[key] = values
        for threshold in config["candidate_thresholds_c"]:
            labels, sizes = component_data(values, threshold, config["connectivity"])
            labels_by_key[(row["label"], row["stem"], threshold)] = labels
            for minimum_area in config["minimum_component_areas"]:
                metrics.append(metrics_for_filter(row["label"], row["stem"], threshold, minimum_area, labels, sizes))

    write_csv(OUTPUT_DIR / "component_filter_metrics.csv", metrics)
    summary = aggregate(metrics, config["candidate_thresholds_c"], config["minimum_component_areas"])
    write_csv(OUTPUT_DIR / "component_filter_summary.csv", summary)
    selections = select_cases(rows, metrics, labels_by_key, config)
    for row, item, labels, name in selections:
        render_case(row, values_by_key[(row["label"], row["stem"])], labels, item, name)

    recommendations = [
        {
            "threshold_c": 60,
            "minimum_component_area": 100,
            "reason": "Retained components occur in 610/622 Fire samples and 0/116 No Fire samples; this is a stricter spatial candidate that should be inspected for lost weak regions.",
        },
        {
            "threshold_c": 80,
            "minimum_component_area": 50,
            "reason": "Retained components occur in 612/622 Fire samples and 0/116 No Fire samples; this balances a higher thermal threshold with a moderate area filter.",
        },
        {
            "threshold_c": 100,
            "minimum_component_area": 10,
            "reason": "Retained components occur in 620/622 Fire samples and 0/116 No Fire samples; this emphasizes intense regions while applying only a small-component cleanup.",
        },
    ]
    report = {
        "config": str(CONFIG_PATH),
        "manifest": str(MANIFEST_PATH),
        "sample_count": len(rows),
        "connectivity": config["connectivity"],
        "candidate_thresholds_c": list(config["candidate_thresholds_c"]),
        "minimum_component_areas": list(config["minimum_component_areas"]),
        "metric_rows": len(metrics),
        "visualization_count": len(selections),
        "pilot_candidates": recommendations,
        "boundary": "These are pilot candidates only. No threshold or component filter is selected, hard-coded as a label rule, or used to generate SAM masks.",
    }
    (OUTPUT_DIR / "component_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()