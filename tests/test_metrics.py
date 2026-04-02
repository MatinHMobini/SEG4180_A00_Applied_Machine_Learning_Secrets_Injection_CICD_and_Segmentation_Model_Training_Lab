import torch
from metrics import dice_score, iou_score

def test_perfect_scores():
    preds = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
    targets = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
    assert abs(dice_score(preds, targets).item() - 1.0) < 1e-6
    assert abs(iou_score(preds, targets).item() - 1.0) < 1e-6