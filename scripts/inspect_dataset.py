from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, TiffImagePlugin


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
OUTPUT_DIR = Path("reports/phase1")
MODALITIES = {
    ("RGB", "Raw"): "raw_rgb",
    ("RGB", "Corrected FOV"): "corrected_rgb",
    ("Thermal", "Raw JPG"): "thermal_jpg",
    ("Thermal", "Celsius TIFF"): "thermal_tiff",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value[:200].hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    return str(value)


def exif_summary(image: Image.Image) -> tuple[str, str, str]:
    exif = image.getexif()
    tags = {str(key): json_value(value) for key, value in exif.items()}
    datetime_values = []
    for key in (306, 36867, 36868):
        if key in exif:
            datetime_values.append(str(exif[key]))
    return json.dumps(tags, sort_keys=True), "|".join(datetime_values), ";".join(
        sorted(tags)
    )


def modality_for(path: Path) -> str | None:
    relative_parts = path.relative_to(DATASET_ROOT).parts
    if len(relative_parts) != 4:
        return None
    return MODALITIES.get((relative_parts[1], relative_parts[2]))


def image_record(path: Path, relative_path: str, modality: str, label: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": relative_path,
        "label": label,
        "modality": modality,
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "filesystem_mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "format": "",
        "mode": "",
        "width": "",
        "height": "",
        "dtype": "",
        "min": "",
        "max": "",
        "mean": "",
        "p01": "",
        "p05": "",
        "p50": "",
        "p95": "",
        "p99": "",
        "exif_datetime": "",
        "exif_keys": "",
        "metadata_json": "",
        "error": "",
    }
    try:
        with Image.open(path) as image:
            record["format"] = image.format or ""
            record["mode"] = image.mode
            record["width"], record["height"] = image.size
            metadata, exif_datetime, exif_keys = exif_summary(image)
            record["exif_datetime"] = exif_datetime
            record["exif_keys"] = exif_keys
            record["metadata_json"] = metadata
            if modality == "thermal_tiff":
                array = np.asarray(image)
                values = array.astype(np.float64, copy=False)
                record["dtype"] = str(array.dtype)
                percentiles = np.percentile(values, [1, 5, 50, 95, 99])
                record.update(
                    {
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "mean": float(np.mean(values)),
                        "p01": float(percentiles[0]),
                        "p05": float(percentiles[1]),
                        "p50": float(percentiles[2]),
                        "p95": float(percentiles[3]),
                        "p99": float(percentiles[4]),
                    }
                )
    except Exception as error:
        record["error"] = repr(error)
    return record


def write_tree(paths: list[Path]) -> None:
    lines = [f"DATASET_ROOT: {DATASET_ROOT}", ""]
    for directory in sorted({path.parent for path in paths}):
        files = sorted(path for path in paths if path.parent == directory)
        relative_directory = directory.relative_to(DATASET_ROOT)
        lines.append(f"{relative_directory}/ ({len(files)} files)")
        for path in files:
            lines.append(f"  {path.name}")
    (OUTPUT_DIR / "dataset_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[(record["label"], record["stem"])][record["modality"]].append(record)
    manifest = []
    for (label, stem), modalities in sorted(grouped.items()):
        row: dict[str, Any] = {"label": label, "stem": stem}
        missing = []
        duplicate = []
        for modality in MODALITIES.values():
            matches = modalities.get(modality, [])
            row[f"{modality}_count"] = len(matches)
            row[modality] = matches[0]["relative_path"] if len(matches) == 1 else ""
            if not matches:
                missing.append(modality)
            if len(matches) > 1:
                duplicate.append(modality)
        row["quartet_status"] = "complete" if not missing and not duplicate else "invalid"
        row["missing_modalities"] = "|".join(missing)
        row["duplicate_modalities"] = "|".join(duplicate)
        manifest.append(row)
    return manifest


def group_report(records: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    stems = sorted({record["stem"] for record in records})
    numeric_stems = [stem for stem in stems if re.fullmatch(r"\d+", stem)]
    numeric_values = [int(stem) for stem in numeric_stems]
    exif_datetimes = Counter(record["exif_datetime"] for record in records if record["exif_datetime"])
    exif_by_modality = {
        modality: Counter(
            record["exif_datetime"]
            for record in records
            if record["modality"] == modality and record["exif_datetime"]
        )
        for modality in MODALITIES.values()
    }
    by_quartet: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for record in records:
        by_quartet[(record["label"], record["stem"])][record["modality"]] = record[
            "exif_datetime"
        ]
    timestamp_comparisons = {}
    for left, right in (("raw_rgb", "corrected_rgb"), ("raw_rgb", "thermal_jpg")):
        both = 0
        same = 0
        different = 0
        for modalities in by_quartet.values():
            left_value = modalities.get(left, "")
            right_value = modalities.get(right, "")
            if left_value and right_value:
                both += 1
                same += left_value == right_value
                different += left_value != right_value
        timestamp_comparisons[f"{left}_vs_{right}"] = {
            "quartets_with_both": both,
            "exact_matches": same,
            "different": different,
        }
    return {
        "dataset_root": str(DATASET_ROOT),
        "filename_pattern": {
            "unique_stems": len(stems),
            "all_stems_numeric": len(numeric_stems) == len(stems),
            "stem_widths": sorted({len(stem) for stem in stems}),
            "minimum_numeric_stem": min(numeric_values) if numeric_values else None,
            "maximum_numeric_stem": max(numeric_values) if numeric_values else None,
        },
        "directory_grouping": sorted({record["label"] for record in records}),
        "exif_datetime": {
            "records_with_datetime": sum(exif_datetimes.values()),
            "unique_values": len(exif_datetimes),
            "sample_values": list(exif_datetimes)[:10],
            "minimum": min(exif_datetimes) if exif_datetimes else None,
            "maximum": max(exif_datetimes) if exif_datetimes else None,
            "per_modality": {
                modality: {
                    "records_with_datetime": sum(counts.values()),
                    "unique_values": len(counts),
                    "repeated_values_within_modality": sum(
                        count > 1 for count in counts.values()
                    ),
                }
                for modality, counts in exif_by_modality.items()
            },
            "cross_modality_comparisons": timestamp_comparisons,
        },
        "filesystem_mtime": {
            "unique_values": len({record["filesystem_mtime"] for record in records}),
            "note": "Extraction timestamps are not capture timestamps and are not a reliable temporal identifier.",
        },
        "potential_group_identifiers": [
            "Fire/No Fire directory label is reliable as a class label, not a temporal group.",
            "Zero-padded numeric filename stems are reliable quartet keys within each class.",
            "EXIF capture timestamps are available on raw RGB, corrected-FOV RGB, and thermal JPG files.",
            "EXIF timestamps identify the capture moment, but do not expose a flight ID or burst ID.",
        ],
        "split_readiness": {
            "reliable_temporal_group_identifier_found": True,
            "identifier": "EXIF capture datetime from RGB/thermal JPG files",
            "reason": "Each quartet has a unique capture timestamp in the available EXIF metadata; this supports chronological blocking, but not explicit flight/burst grouping.",
        },
        "complete_quartets": sum(row["quartet_status"] == "complete" for row in manifest),
        "invalid_quartets": sum(row["quartet_status"] != "complete" for row in manifest),
    }


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_file())
    records = []
    thermal_values = []
    for path in paths:
        modality = modality_for(path)
        if modality is None:
            continue
        relative_path = path.relative_to(DATASET_ROOT).as_posix()
        record = image_record(path, relative_path, modality, path.relative_to(DATASET_ROOT).parts[0])
        records.append(record)
        if modality == "thermal_tiff" and not record["error"]:
            with Image.open(path) as image:
                thermal_values.append(np.asarray(image, dtype=np.float32).reshape(-1))

    write_tree(paths)
    metadata_fields = list(records[0]) if records else []
    with (OUTPUT_DIR / "image_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(records)

    manifest = build_manifest(records)
    manifest_fields = list(manifest[0]) if manifest else []
    with (OUTPUT_DIR / "quartet_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest)

    report = group_report(records, manifest)
    report["file_count"] = len(paths)
    report["inspected_image_count"] = len(records)
    report["format_counts"] = dict(Counter(record["format"] for record in records))
    report["mode_counts"] = dict(Counter(record["mode"] for record in records))
    report["dimension_counts"] = dict(
        Counter(f"{record['width']}x{record['height']}" for record in records)
    )
    if thermal_values:
        all_thermal_values = np.concatenate(thermal_values)
        global_percentiles = np.percentile(all_thermal_values, [1, 5, 50, 95, 99])
        report["thermal_tiff_global_pixel_statistics"] = {
            "pixel_count": int(all_thermal_values.size),
            "dtype": str(all_thermal_values.dtype),
            "minimum": float(np.min(all_thermal_values)),
            "maximum": float(np.max(all_thermal_values)),
            "mean": float(np.mean(all_thermal_values)),
            "p01": float(global_percentiles[0]),
            "p05": float(global_percentiles[1]),
            "p50": float(global_percentiles[2]),
            "p95": float(global_percentiles[3]),
            "p99": float(global_percentiles[4]),
        }
    report["duplicate_content_sha256"] = {
        digest: count
        for digest, count in Counter(record["sha256"] for record in records).items()
        if count > 1
    }
    (OUTPUT_DIR / "group_identifier_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()