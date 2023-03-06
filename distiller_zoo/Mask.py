import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        # self.kl = DistillKL(4)

    def forward(self, feat_s, feat_t, mask):
        n = 1
        loss = torch.tensor(0.).cuda()
        # kl_loss = torch.tensor(0.).cuda()
        for i in range(len(feat_s) - 1, -1, -1):
            bs, c, h, w = feat_s[i].shape
            # if i == len(feat_s) - 1:
            #     mask = torch.where(feat_t[i].sum(1, keepdim=True) <= 0, torch.tensor(0.5).cuda(), torch.tensor(1.0).cuda())
            # else:
            #    mask = F.interpolate(mask, (h, w), mode='nearest')
            # mask = torch.where(feat_t[i] <= 0, torch.tensor(0.5).cuda(), torch.tensor(1.0).cuda())
            mse = self.mse(feat_s[i], feat_t[i])
            # kl_loss += self.kl(feat_s[i], feat_t[i])
            loss += ((mse * mask[i]).sum() / (1.0 * bs * n * h * w))
            n *= 4

        return loss