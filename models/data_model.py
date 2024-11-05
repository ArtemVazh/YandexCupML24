from typing import TypedDict

import torch


class ValDict(TypedDict):
    anchor_id: int
    f_t: torch.Tensor
    f_c: torch.Tensor


class BatchDict(TypedDict):
    anchor_id: int
    anchor: torch.Tensor
    anchor_label: torch.Tensor
    positive_id: int
    positive: torch.Tensor
    negative_id: int
    negative: torch.Tensor


class Postfix(TypedDict):
    Epoch: int
    train_loss: float
    train_loss_step: float
    train_cls_loss: float
    train_cls_loss_step: float
    train_triplet_loss: float
    train_triplet_loss_step: float
    val_loss: float
    mr1: float
    mAP: float


class TestResults(TypedDict):
    test_mr1: float
    test_mAP: float
