from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
MANIFEST_PATH = Path("reports/phase1/quartet_manifest.csv")
OUTPUT_DIR = Path("reports/phase2")
CONFIG_PATH = Path("configs/thermal_analysis.json")


def load_config() -> tuple[tuple[float, ...], int]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = tuple(float(value) for value in config["candidate_thresholds_c"])
    connectivity = int(config["connectivity"])
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    return thresholds, connectivity


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    invalid = [row["stem"] for row in rows if row["quartet_status"] != "complete"]
    if invalid:
        raise ValueError(f"Manifest contains invalid quartets: {invalid[:5]}")
    return rows


def load_temperature(row: dict[str, str]) -> np.ndarray:
    path = DATASET_ROOT / row["thermal_tiff"]
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    if values.shape != (512, 640):
        raise ValueError(f"Unexpected TIFF shape for {path}: {values.shape}")
    return values


def distribution_record(row: dict[str, str], values: np.ndarray) -> dict[str, float | str]:
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "label": row["label"],
        "stem": row["stem"],
        "minimum_c": float(np.min(values)),
        "maximum_c": float(np.max(values)),
        "mean_c": float(np.mean(values)),
        "std_c": float(np.std(values)),
        "p01_c": float(percentiles[0]),
        "p05_c": float(percentiles[1]),
        "p25_c": float(percentiles[2]),
        "p50_c": float(percentiles[3]),
        "p75_c": float(percentiles[4]),
        "p95_c": float(percentiles[5]),
        "p99_c": float(percentiles[6]),
    }


def threshold_records(
    row: dict[str, str], values: np.ndarray, thresholds: tuple[float, ...], connectivity_value: int
) -> list[dict[str, float | str]]:
    pixel_count = values.size
    records = []
    connectivity = np.ones((3, 3), dtype=np.uint8) if connectivity_value == 8 else ndimage.generate_binary_structure(2, 1)
    for threshold in thresholds:
        above = values >= threshold
        labeled, component_count = ndimage.label(above, structure=connectivity)
        foreground_pixels = int(np.count_nonzero(above))
        largest_area = 0
        largest_label = 0
        if component_count:
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_label = int(np.argmax(component_sizes)) + 1
            largest_area = int(component_sizes[largest_label - 1])
        if largest_label:
            coordinates = np.argwhere(labeled == largest_label)
            y_min, x_min = coordinates.min(axis=0)
            y_max, x_max = coordinates.max(axis=0)
            x_min, y_min, x_max, y_max = (
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max),
            )
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            bbox_area = bbox_width * bbox_height
        else:
            x_min = y_min = x_max = y_max = 0
            bbox_width = bbox_height = bbox_area = 0
        records.append(
            {
                "label": row["label"],
                "stem": row["stem"],
                "threshold_c": threshold,
                "any_pixels_exceed": bool(foreground_pixels > 0),
                "connected_components": int(component_count),
                "largest_component_area_pixels": largest_area,
                "largest_component_percent_image": float(largest_area * 100 / pixel_count),
                "foreground_pixels": foreground_pixels,
                "foreground_percent_image": float(foreground_pixels * 100 / pixel_count),
                "bbox_x_min": x_min,
                "bbox_y_min": y_min,
                "bbox_x_max": x_max,
                "bbox_y_max": y_max,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area_pixels": bbox_area,
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_summary(distributions: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    numeric_fields = [
        "minimum_c",
        "maximum_c",
        "mean_c",
        "std_c",
        "p01_c",
        "p05_c",
        "p25_c",
        "p50_c",
        "p75_c",
        "p95_c",
        "p99_c",
    ]
    for label in ("Fire", "No Fire"):
        selected = [row for row in distributions if row["label"] == label]
        summary[label] = {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in numeric_fields
        }
        summary[label]["sample_count"] = float(len(selected))
    return summary


def threshold_summary(records: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows = []
    thresholds = sorted({float(row["threshold_c"]) for row in records})
    for threshold in thresholds:
        row: dict[str, float | str] = {"threshold_c": threshold}
        for label in ("Fire", "No Fire"):
            selected = [
                float(item["foreground_percent_image"])
                for item in records
                if item["label"] == label and item["threshold_c"] == threshold
            ]
            row[f"{label.lower().replace(' ', '_')}_mean_percent_above"] = float(np.mean(selected))
            row[f"{label.lower().replace(' ', '_')}_median_percent_above"] = float(np.median(selected))
            row[f"{label.lower().replace(' ', '_')}_min_percent_above"] = float(np.min(selected))
            row[f"{label.lower().replace(' ', '_')}_max_percent_above"] = float(np.max(selected))
            component_values = [
                int(item["connected_components"])
                for item in records
                if item["label"] == label and item["threshold_c"] == threshold
            ]
            largest_values = [
                float(item["largest_component_percent_image"])
                for item in records
                if item["label"] == label and item["threshold_c"] == threshold
            ]
            row[f"{label.lower().replace(' ', '_')}_mean_components"] = float(np.mean(component_values))
            row[f"{label.lower().replace(' ', '_')}_median_components"] = float(np.median(component_values))
            row[f"{label.lower().replace(' ', '_')}_mean_largest_component_percent"] = float(np.mean(largest_values))
            row[f"{label.lower().replace(' ', '_')}_median_largest_component_percent"] = float(np.median(largest_values))
        rows.append(row)
    return rows


def render_sample(
    row: dict[str, str],
    values: np.ndarray,
    distribution: dict[str, object],
    threshold: float,
    output_name: str,
) -> None:
    rgb_path = DATASET_ROOT / row["corrected_rgb"]
    thermal_path = DATASET_ROOT / row["thermal_jpg"]
    with Image.open(rgb_path) as rgb_image, Image.open(thermal_path) as thermal_image:
        rgb = np.asarray(rgb_image.convert("RGB"))
        thermal = np.asarray(thermal_image.convert("RGB"))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(f"{row['label']} sample {row['stem']} | corrected RGB and radiometric thermal")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Corrected-FOV RGB")
    axes[0, 1].imshow(thermal)
    axes[0, 1].set_title("Thermal JPG")
    image = axes[1, 0].imshow(values, cmap="inferno")
    axes[1, 0].set_title("Radiometric temperature (C)")
    axes[1, 0].set_xlabel(
        f"min={distribution['minimum_c']:.1f}, max={distribution['maximum_c']:.1f}, "
        f"mean={distribution['mean_c']:.1f}"
    )
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 1].hist(values.ravel(), bins=80, color="#d95f02", alpha=0.9)
    axes[1, 1].axvline(threshold, color="black", linestyle="--", label=f"{threshold:g} C")
    axes[1, 1].legend()
    axes[1, 1].set_title("Temperature histogram")
    axes[1, 1].set_xlabel("Temperature (C)")
    axes[1, 1].set_ylabel("Pixel count")
    for axis in axes.flat:
        axis.set_xticks([]) if axis in (axes[0, 0], axes[0, 1]) else None
        axis.set_yticks([]) if axis in (axes[0, 0], axes[0, 1]) else None
    output_path = OUTPUT_DIR / "visualizations" / f"{output_name}.png"
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def edge_case_selections(
    rows: list[dict[str, str]],
    spatial_rows: list[dict[str, float | str]],
    values_by_key: dict[tuple[str, str], np.ndarray],
    thresholds: tuple[float, ...],
) -> list[tuple[dict[str, str], float, str]]:
    by_key = {(row["label"], row["stem"]): row for row in rows}
    selections: list[tuple[dict[str, str], float, str]] = []
    for label, prefix in (("Fire", "fire"), ("No Fire", "no_fire")):
        threshold_rows = [row for row in spatial_rows if row["label"] == label]
        for threshold in thresholds:
            selected = [row for row in threshold_rows if row["threshold_c"] == threshold]
            if label == "Fire":
                small = min(selected, key=lambda item: int(item["largest_component_area_pixels"]))
                large = max(selected, key=lambda item: int(item["largest_component_area_pixels"]))
                for item, name in ((small, "smallest"), (large, "largest")):
                    selections.append((by_key[(label, item["stem"])], threshold, f"{prefix}_{name}_{int(threshold)}c_{item['stem']}"))
            else:
                largest = max(selected, key=lambda item: int(item["largest_component_area_pixels"]))
                selections.append((by_key[(label, largest["stem"])], threshold, f"{prefix}_largest_{int(threshold)}c_{largest['stem']}"))

    threshold_ranges = {}
    for key in values_by_key:
        candidates = [row for row in spatial_rows if (row["label"], row["stem"]) == key]
        foreground = [float(row["foreground_percent_image"]) for row in candidates]
        threshold_ranges[key] = max(foreground) - min(foreground)
    for key, _range in sorted(threshold_ranges.items(), key=lambda item: item[1], reverse=True)[:6]:
        candidates = [row for row in spatial_rows if (row["label"], row["stem"]) == key]
        item = max(candidates, key=lambda row: float(row["foreground_percent_image"]))
        row = by_key[key]
        selections.append((row, float(item["threshold_c"]), f"threshold_sensitive_{row['label'].lower().replace(' ', '_')}_{row['stem']}"))
    unique = {}
    for row, threshold, name in selections:
        unique[name] = (row, threshold, name)
    return list(unique.values())


def main() -> None:
    thresholds, connectivity = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)
    rows = read_manifest()
    distributions = []
    threshold_rows = []
    values_by_key: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        values = load_temperature(row)
        values_by_key[(row["label"], row["stem"])] = values
        distributions.append(distribution_record(row, values))
        threshold_rows.extend(threshold_records(row, values, thresholds, connectivity))

    write_csv(OUTPUT_DIR / "temperature_distributions.csv", distributions)
    write_csv(OUTPUT_DIR / "threshold_spatial_metrics.csv", threshold_rows)
    write_csv(OUTPUT_DIR / "threshold_pixel_percentages.csv", threshold_rows)
    summary_rows = threshold_summary(threshold_rows)
    write_csv(OUTPUT_DIR / "threshold_summary.csv", summary_rows)

    distribution_by_key = {
        (row["label"], row["stem"]): distribution
        for row, distribution in zip(rows, distributions)
    }
    selections = edge_case_selections(rows, threshold_rows, values_by_key, thresholds)
    for row, threshold, name in selections:
        render_sample(
            row,
            values_by_key[(row["label"], row["stem"])],
            distribution_by_key[(row["label"], row["stem"])],
            threshold,
            name,
        )

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "manifest": str(MANIFEST_PATH),
        "config": str(CONFIG_PATH),
        "sample_counts": {label: sum(row["label"] == label for row in rows) for label in ("Fire", "No Fire")},
        "candidate_thresholds_c": list(thresholds),
        "connectivity": connectivity,
        "class_mean_distribution_statistics": class_summary(distributions),
        "threshold_summary": summary_rows,
        "visualization_samples": [
            {"label": row["label"], "stem": row["stem"], "threshold_c": threshold, "name": name}
            for row, threshold, name in selections
        ],
        "interpretation_boundary": (
            "Threshold statistics quantify hot-pixel coverage only. They do not create labels, "
            "connected components, bounding boxes, or SAM masks. Spatial plausibility requires "
            "visual review of the saved temperature maps and histograms."
        ),
    }
    (OUTPUT_DIR / "thermal_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()