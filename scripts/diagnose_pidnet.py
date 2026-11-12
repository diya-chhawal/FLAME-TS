import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.pidnet import get_seg_model
from scripts.flame3_pidnet_dataset import FLAME3SegmentationDataset


dataset = FLAME3SegmentationDataset(
    "reports/phase3_split/split_manifest.csv",
    "reports/phase3_pseudolabels/full",
    "validation",
)


class Config:
    pass


cfg = Config()
cfg.MODEL = Config()
cfg.MODEL.NAME = "pidnet_s"

model = get_seg_model(
    cfg=cfg,
    imgnet_pretrained=False,
).cuda()

checkpoint = torch.load(
    "checkpoints/pidnet_flame3/best.pt",
    map_location="cuda",
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

thresholds = [0.3, 0.5, 0.7, 0.9]

stats = {
    t: {"pred_fg": [], "iou": [], "dice": [], "zero_pred": 0}
    for t in thresholds
}

with torch.no_grad():
    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample["image"].unsqueeze(0).cuda()
        target = sample["mask"].unsqueeze(0).cuda()

        logits = model(image)[1]

        logits = torch.nn.functional.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        probabilities = torch.sigmoid(logits)

        for threshold in thresholds:
            prediction = (probabilities >= threshold).float()

            intersection = (prediction * target).sum()
            union = ((prediction + target) > 0).float().sum()

            pred_sum = prediction.sum()
            target_sum = target.sum()

            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union

            denominator = pred_sum + target_sum

            if denominator == 0:
                dice = 1.0
            else:
                dice = 2.0 * intersection / denominator

            pred_fg = prediction.mean().item()

            stats[threshold]["pred_fg"].append(pred_fg)
            stats[threshold]["iou"].append(iou.item())
            stats[threshold]["dice"].append(dice.item())

            if pred_sum.item() == 0:
                stats[threshold]["zero_pred"] += 1


print("validation samples:", len(dataset))
print()

mean_target_fg = sum(
    dataset[i]["mask"].mean().item()
    for i in range(len(dataset))
) / len(dataset)

print("mean target foreground:", mean_target_fg)
print()

for threshold in thresholds:
    pred_fg = sum(stats[threshold]["pred_fg"]) / len(dataset)
    iou = sum(stats[threshold]["iou"]) / len(dataset)
    dice = sum(stats[threshold]["dice"]) / len(dataset)
    zero_pred = stats[threshold]["zero_pred"]

    print(f"threshold: {threshold}")
    print(f"  mean predicted foreground: {pred_fg}")
    print(f"  mean IoU: {iou}")
    print(f"  mean Dice: {dice}")
    print(f"  zero-prediction samples: {zero_pred}/{len(dataset)}")