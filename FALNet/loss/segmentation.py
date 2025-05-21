import torch
import torch.nn.functional as F
import torch.nn as nn

def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)
    adice = 1 - (2 * inter + 1) / (union + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    return (0.7 * abce + 0.7 * aiou + 0.7 * adice + 0.7 * amae).mean()

   
class CAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = adaptive_pixel_intensity_loss

    def forward(self, cam, masks):
        ds_cam = F.interpolate(cam, scale_factor=4, mode='bilinear', align_corners=True)
        return self.criterion(torch.sigmoid(ds_cam), masks)

