from pathlib import Path
import pandas as pd
from itertools import permutations

MANIFEST = Path("reports/phase1/image_metadata.csv")
TARGET = (0.70, 0.15, 0.15)

df = pd.read_csv(MANIFEST)
df = df[df["modality"] == "corrected_rgb"].copy()
df["time"] = pd.to_datetime(
    df["exif_datetime"], format="%Y:%m:%d %H:%M:%S"
)
df = df.sort_values("time").reset_index(drop=True)

# Temporal blocks: a new block starts after >=60 seconds.
gap = df["time"].diff().dt.total_seconds()
df["block"] = (gap >= 60).cumsum()

blocks = []

for bid, g in df.groupby("block", sort=True):
    blocks.append({
        "id": int(bid) + 1,
        "n": len(g),
        "fire": int((g["label"] == "Fire").sum()),
        "nofire": int((g["label"] == "No Fire").sum()),
        "df": g,
    })

print("Blocks:")
for b in blocks:
    print(
        f"B{b['id']:02d}: n={b['n']:3d}, "
        f"Fire={b['fire']:3d}, NoFire={b['nofire']:3d}"
    )

# ---------------------------------------------------------
# Block 2: create contiguous label runs.
# ---------------------------------------------------------
B2 = next(b for b in blocks if b["id"] == 2)

run_id = (
    B2["df"]["label"] != B2["df"]["label"].shift()
).cumsum()

runs = []

for _, g in B2["df"].groupby(run_id):
    runs.append({
        "label": g["label"].iloc[0],
        "n": len(g),
        "fire": int((g["label"] == "Fire").sum()),
        "nofire": int((g["label"] == "No Fire").sum()),
        "df": g,
    })

print("\nBlock 2 runs:")
for i, r in enumerate(runs, 1):
    print(
        f"R{i:02d}: {r['label']:7s} n={r['n']:3d}"
    )

# ---------------------------------------------------------
# Generate every valid 3-chunk split of Block 2.
# Cuts may only occur between label runs.
# Each chunk must contain both classes.
# ---------------------------------------------------------
b2_candidates = []

for cut1 in range(1, len(runs) - 1):
    for cut2 in range(cut1 + 1, len(runs)):

        raw_chunks = [
            runs[:cut1],
            runs[cut1:cut2],
            runs[cut2:],
        ]

        chunks = []

        for chunk in raw_chunks:
            g = pd.concat([r["df"] for r in chunk])

            chunks.append({
                "name": f"B2[{chunk[0]['label']}...{chunk[-1]['label']}]",
                "n": len(g),
                "fire": int((g["label"] == "Fire").sum()),
                "nofire": int((g["label"] == "No Fire").sum()),
                "df": g,
            })

        if all(
            c["fire"] > 0 and c["nofire"] > 0
            for c in chunks
        ):
            b2_candidates.append(chunks)

print(
    f"\nValid Block-2 three-way cuts: "
    f"{len(b2_candidates)}"
)

# ---------------------------------------------------------
# B7 is also mixed and remains indivisible.
# All other blocks are Fire-only and indivisible.
# ---------------------------------------------------------
B7 = next(b for b in blocks if b["id"] == 7)

pure_blocks = [
    b for b in blocks
    if b["id"] not in (2, 7)
]

print("\nPure Fire-only blocks:")
print([f"B{b['id']:02d}" for b in pure_blocks])

TOTAL = len(df)
target = [TOTAL * x for x in TARGET]

best = None


def score_sizes(sizes):
    ratios = [x / TOTAL for x in sizes]
    return sum(
        abs(r - t)
        for r, t in zip(ratios, TARGET)
    )


# ---------------------------------------------------------
# DP over Fire-only blocks.
#
# State = (train_count, val_count)
# Test count is determined by total.
#
# For every reachable state we store one assignment.
# ---------------------------------------------------------
def build_pure_dp():

    states = {
        (0, 0): []
    }

    for b in pure_blocks:

        new_states = {}

        for (train_n, val_n), assignment in states.items():

            # Put block in Train.
            key = (
                train_n + b["n"],
                val_n,
            )

            if key not in new_states:
                new_states[key] = assignment + [("Train", b)]

            # Put block in Validation.
            key = (
                train_n,
                val_n + b["n"],
            )

            if key not in new_states:
                new_states[key] = assignment + [("Validation", b)]

            # Put block in Test.
            key = (
                train_n,
                val_n,
            )

            if key not in new_states:
                new_states[key] = assignment + [("Test", b)]

        states = new_states

    return states


print("\nBuilding DP table...")
pure_states = build_pure_dp()
print(f"Reachable DP states: {len(pure_states)}")


# ---------------------------------------------------------
# Combine:
#   - one valid B2 3-way split
#   - B7
#   - all Fire-only blocks
#
# Each split must receive at least one mixed unit.
# This guarantees Fire + NoFire in every split.
# ---------------------------------------------------------
for candidate_index, chunks in enumerate(b2_candidates, 1):

    # Assign the 3 B2 chunks to Train/Val/Test.
    for perm in permutations(range(3)):

        base = [
            {"n": 0, "fire": 0, "nofire": 0, "units": []}
            for _ in range(3)
        ]

        for chunk_index, split in enumerate(perm):
            c = chunks[chunk_index]

            base[split]["n"] += c["n"]
            base[split]["fire"] += c["fire"]
            base[split]["nofire"] += c["nofire"]
            base[split]["units"].append(
                ("B2 chunk", c)
            )

        # Try every possible destination for B7.
        for b7_split in range(3):

            mixed = [
                {
                    "n": x["n"],
                    "fire": x["fire"],
                    "nofire": x["nofire"],
                    "units": list(x["units"]),
                }
                for x in base
            ]

            mixed[b7_split]["n"] += B7["n"]
            mixed[b7_split]["fire"] += B7["fire"]
            mixed[b7_split]["nofire"] += B7["nofire"]
            mixed[b7_split]["units"].append(
                ("B7", B7)
            )

            # Every split must already have some NoFire.
            if any(x["nofire"] == 0 for x in mixed):
                continue

            for (train_pure, val_pure), assignment in pure_states.items():

                pure_total = sum(b["n"] for b in pure_blocks)

                test_pure = (
                    pure_total
                    - train_pure
                    - val_pure
                )

                sizes = [
                    mixed[0]["n"] + train_pure,
                    mixed[1]["n"] + val_pure,
                    mixed[2]["n"] + test_pure,
                ]

                score = score_sizes(sizes)

                if best is None or score < best["score"]:

                    # Reconstruct pure assignment.
                    groups = [
                        list(mixed[0]["units"]),
                        list(mixed[1]["units"]),
                        list(mixed[2]["units"]),
                    ]

                    for split_name, b in assignment:
                        split_index = {
                            "Train": 0,
                            "Validation": 1,
                            "Test": 2,
                        }[split_name]

                        groups[split_index].append(
                            (f"B{b['id']:02d}", b)
                        )

                    counts = []

                    for group in groups:
                        n = sum(x[1]["n"] for x in group)
                        fire = sum(x[1]["fire"] for x in group)
                        nofire = sum(
                            x[1]["nofire"] for x in group
                        )

                        counts.append(
                            (n, fire, nofire)
                        )

                    best = {
                        "score": score,
                        "counts": counts,
                        "groups": groups,
                        "candidate": candidate_index,
                    }


# ---------------------------------------------------------
# Report result.
# ---------------------------------------------------------
print("\nBEST SPLIT")
print("==========")

if best is None:
    print("No valid split found.")
    raise SystemExit(1)

print(f"Score: {best['score']:.6f}")
print(f"Block-2 candidate: {best['candidate']}")

for name, (n, fire, nofire) in zip(
    ["Train", "Validation", "Test"],
    best["counts"],
):
    print(
        f"{name:10s}: "
        f"n={n:3d}, "
        f"Fire={fire:3d}, "
        f"NoFire={nofire:3d}, "
        f"ratio={n / TOTAL:.3f}"
    )

print("\nAssigned units:")

for name, group in zip(
    ["Train", "Validation", "Test"],
    best["groups"],
):
    print(f"\n{name}:")

    for label, item in group:
        print(
            f"  {label}: "
            f"n={item['n']:3d}, "
            f"Fire={item['fire']:3d}, "
            f"NoFire={item['nofire']:3d}"
        )