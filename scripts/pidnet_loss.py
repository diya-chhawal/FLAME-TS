import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce(bd_pre, target):
    """
    Class-balanced BCE used by the released WACV PIDNet code.
    """
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = target_t == 1
    neg_index = target_t == 0

    weight = torch.zeros_like(log_p)

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    if sum_num == 0:
        return log_p.sum() * 0.0

    weight[pos_index] = neg_num.float() / sum_num.float()
    weight[neg_index] = pos_num.float() / sum_num.float()

    return F.binary_cross_entropy_with_logits(
        log_p,
        target_t,
        weight=weight,
        reduction="mean",
    )


class BoundaryLoss(nn.Module):
    """
    Released WACV boundary loss:
        20 * weighted_bce(...)
    """

    def __init__(self, coeff_bce=20.0):
        super().__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        return self.coeff_bce * weighted_bce(bd_pre, bd_gt)


class PIDNetLoss(nn.Module):
    """
    Loss matching the released WACV FullModel combination.

    outputs:
        [auxiliary_segmentation,
         main_segmentation,
         boundary_prediction]
    """

    def __init__(self, pos_weight=2.5):
        super().__init__()

        self.sem_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        self.bd_loss = BoundaryLoss()

    def forward(self, outputs, target, boundary_target):
        if len(outputs) != 3:
            raise ValueError(
                f"Expected 3 PIDNet training outputs, got {len(outputs)}"
            )

        aux_output, main_output, boundary_output = outputs

        # Ensure all predictions have the same spatial resolution
        # as the FLAME 3 targets.
        target_size = target.shape[-2:]

        aux_output = F.interpolate(
            aux_output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        main_output = F.interpolate(
            main_output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        boundary_output = F.interpolate(
            boundary_output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        loss_s0 = self.sem_loss(aux_output, target)
        loss_s = self.sem_loss(main_output, target)
        loss_b = self.bd_loss(boundary_output, boundary_target)

        # Boundary-aware semantic term from the released WACV code.
        boundary_mask = (
            torch.sigmoid(boundary_output) > 0.8
        ).float()

        filler = torch.zeros_like(target)

        boundary_label = torch.where(
            boundary_mask.bool(),
            target,
            filler,
        )

        boundary_prediction = torch.where(
            boundary_mask.bool(),
            main_output,
            filler,
        )

        loss_sb = self.sem_loss(
            boundary_prediction,
            boundary_label,
        )

        total_loss = (
            0.4 * loss_s0
            + 20.0 * loss_s
            + loss_b
            + loss_sb
        )

        return {
            "total": total_loss,
            "seg_aux": loss_s0,
            "seg_main": loss_s,
            "boundary": loss_b,
            "boundary_semantic": loss_sb,
        }
if __name__ == "__main__":
    torch.manual_seed(0)

    criterion = PIDNetLoss()

    outputs = [
        torch.randn(2, 1, 64, 80),
        torch.randn(2, 1, 64, 80),
        torch.randn(2, 1, 64, 80),
    ]

    target = torch.randint(
        0, 2, (2, 1, 512, 640)
    ).float()

    boundary_target = torch.randint(
        0, 2, (2, 1, 512, 640)
    ).float()

    losses = criterion(
        outputs,
        target,
        boundary_target,
    )

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")
