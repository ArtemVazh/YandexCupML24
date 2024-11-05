import numpy as np


class EarlyStopper:
    def __init__(self, patience: int = 1, delta: int = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_validation_mAP = -np.inf

    def __call__(self, validation_mAP) -> bool:
        if validation_mAP > self.max_validation_mAP:
            self.max_validation_mAP = validation_mAP
            self.counter = 0
        elif validation_mAP <= (self.max_validation_mAP - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
