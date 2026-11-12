from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
DEV_FRACTION = 0.15

MANIFEST = Path("reports/phase3_split/split_manifest.csv")
OUT_DIR = Path("reports/phase3_split")

DATA_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
THERMAL_DIR = DATA_ROOT / "Fire" / "Thermal" / "Celsius TIFF"


def thermal_area_percent(stem: int) -> float:
    from PIL import Image

    path = THERMAL_DIR / f"{stem:05d}.tiff"
    thermal = np.asarray(Image.open(path))
    return float((thermal >= 80.0).mean() * 100.0)


def main():
    rng = np.random.default_rng(SEED)

    manifest = pd.read_csv(MANIFEST)

    train = manifest[manifest["split"].eq("train")].copy()

    fire = train[train["label"].eq("Fire")].copy()
    no_fire = train[train["label"].eq("No Fire")].copy()

    fire["thermal_area_percent"] = [
        thermal_area_percent(int(stem)) for stem in fire["stem"]
    ]

    # Five quantile bins, computed only from the current training Fire set.
    fire["thermal_bin"] = pd.qcut(
        fire["thermal_area_percent"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    # Sample approximately 15% from each Fire thermal-area bin.
    dev_fire_parts = []

    for _, group in fire.groupby("thermal_bin", sort=True):
        n_dev = max(1, round(len(group) * DEV_FRACTION))
        indices = rng.choice(group.index.to_numpy(), size=n_dev, replace=False)
        dev_fire_parts.append(group.loc[indices])

    dev_fire = pd.concat(dev_fire_parts).copy()

    # Sample approximately 15% of No Fire.
    n_dev_no_fire = max(1, round(len(no_fire) * DEV_FRACTION))
    no_fire_indices = rng.choice(
        no_fire.index.to_numpy(),
        size=n_dev_no_fire,
        replace=False,
    )
    dev_no_fire = no_fire.loc[no_fire_indices].copy()

    dev = pd.concat([dev_fire, dev_no_fire]).copy()

    # Everything else from the original training split remains training data.
    dev_keys = set(zip(dev["label"], dev["stem"]))

    train_dev = train[
        ~train.apply(lambda r: (r["label"], r["stem"]) in dev_keys, axis=1)
    ].copy()

    dev["split"] = "dev"
    train_dev["split"] = "train_dev"

    # Remove diagnostic-only columns before saving.
    dev = dev.drop(columns=["thermal_area_percent", "thermal_bin"])
    train_dev = train_dev.drop(
        columns=["thermal_area_percent", "thermal_bin"],
        errors="ignore",
    )

    dev = dev.sort_values(["label", "stem"]).reset_index(drop=True)
    train_dev = train_dev.sort_values(["label", "stem"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dev_path = OUT_DIR / "dev_manifest.csv"
    train_dev_path = OUT_DIR / "train_dev_manifest.csv"

    dev.to_csv(dev_path, index=False)
    train_dev.to_csv(train_dev_path, index=False)

    print("Development split created.")
    print()
    print("DEV:")
    print(dev.groupby(["label"]).size())
    print(f"Total: {len(dev)}")
    print()
    print("TRAIN_DEV:")
    print(train_dev.groupby(["label"]).size())
    print(f"Total: {len(train_dev)}")
    print()
    print("Fire thermal-area distribution in DEV:")
    print(
        fire[fire.index.isin(dev_fire.index)]["thermal_area_percent"]
        .describe(percentiles=[0.5, 0.95])
        .round(4)
    )
    print()
    print(f"Saved: {dev_path}")
    print(f"Saved: {train_dev_path}")


if __name__ == "__main__":
    main()