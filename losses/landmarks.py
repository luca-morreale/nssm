
import torch

from .mixin import Loss


class L2Loss(Loss):

    def forward(self, pred, gt, aggregate=False):
        loss = (gt - pred).pow(2).sum(-1).sqrt()

        if aggregate:
            return loss.mean()
        return loss

class L1Loss(Loss):

    def forward(self, pred, gt, aggregate=False):
        loss = (gt - pred).abs().sum(-1)

        if aggregate:
            return loss.mean()
        return loss

class WL1Loss(Loss):

    def forward(self, pred, gt, aggregate=False):
        # with torch.no_grad():
        #     w = 1.0 / ((gt - pred).pow(2).sum(-1) + 1.0e-4)
        # loss = w * (gt - pred).pow(2).sum(-1).sqrt()
        # loss = torch.min((gt - pred).pow(2).sum(-1).sqrt(), torch.ones([1], device=pred.device)*0.2)
        loss = (gt - pred).pow(2).sum(-1).sqrt()
        loss = torch.tanh(5.0 * loss) # clamp the loss to 1


        if aggregate:
            return loss.mean()
        return loss

class LpLoss(Loss):
    p=0.5

    def forward(self, pred, gt, aggregate=False):
        loss = (gt - pred).abs().pow(self.p).sum(-1).pow(1/self.p).clamp(min=0, max=1.0e2)

        if aggregate:
            return loss.mean()
        return loss

class LhalfLoss(LpLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = 0.5
