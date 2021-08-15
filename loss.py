"""
Loss Functions

MIT License
(C) 2021 Athina Lange
"""

import torch.nn as nn


class Loss(nn.Module):
    """ Loss Function
    """
    def __init__(self, loss):
        super(Loss, self).__init__()

        # Loss function
        losses = {"BCE": nn.BCEWithLogitsLoss(reduction='none')}
        self.loss = losses[loss]

    def forward(self, prediction, ground_truth, mask=None):

        # When the model (e.g. no padding) provides predictions that are smaller than the ground-truth
        # center crop ground-truth
        if prediction.shape[2] < ground_truth.shape[2] or \
           prediction.shape[3] < ground_truth.shape[3]:
            y = int((ground_truth.shape[2] - prediction.shape[2]) / 2)
            x = int((ground_truth.shape[3] - prediction.shape[3]) / 2)
            ground_truth = ground_truth[:, :, y:y + prediction.shape[2], x:x + prediction.shape[3]]
            if mask is not None:
                mask = mask[:, :, y:y + prediction.shape[2], x:x + prediction.shape[3]]

        if mask is not None:
            return (self.loss(prediction, ground_truth) * mask).sum() / mask.sum()
        else:
            return self.loss(prediction, ground_truth).mean()
