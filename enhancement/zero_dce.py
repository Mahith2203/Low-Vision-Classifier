"""
Zero-DCE (Zero-Reference Deep Curve Estimation) for low-light image enhancement.
Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" (CVPR 2020)

Zero-DCE learns a set of pixel-wise curve parameter maps and iteratively applies
a light-enhancement curve to the input image. It needs NO paired ground truth data,
only a set of loss functions (spatial consistency, exposure, color constancy, illumination smoothness).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroDCE(nn.Module):
    """
    Lightweight CNN that predicts 8 curve-parameter maps (3 channels each),
    applied iteratively to enhance the low-light image.
    """
    def __init__(self, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations
        num_curve_maps = num_iterations * 3  # 3 channels (RGB) per iteration

        self.relu = nn.ReLU(inplace=True)

        # Simple 7-layer CNN with skip connections (as in the original paper)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)   # concat skip from conv3
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)   # concat skip from conv2
        self.conv7 = nn.Conv2d(64, num_curve_maps, 3, 1, 1)  # concat skip from conv1

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))  # curve params in [-1, 1]

        curve_maps = torch.split(x_r, 3, dim=1)  # list of (B,3,H,W), len = num_iterations

        enhanced = x
        for curve in curve_maps:
            # Light-Enhancement Curve: LE(x) = x + curve * x * (1 - x)
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)

        return enhanced, x_r


# ---------------- Loss functions used to train Zero-DCE (no paired data needed) ----------------

class ColorConstancyLoss(nn.Module):
    """Penalizes deviation between average R, G, B channel intensities (Gray-World assumption)."""
    def forward(self, x):
        mean_rgb = x.mean(dim=[2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        return torch.mean(torch.sqrt(d_rg ** 2 + d_rb ** 2 + d_gb ** 2 + 1e-8))


class ExposureLoss(nn.Module):
    """Controls exposure level, encouraging avg intensity in local patches towards a target (e.g. 0.6)."""
    def __init__(self, patch_size=16, mean_val=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        mean_patches = self.pool(x_gray)
        return torch.mean(torch.pow(mean_patches - self.mean_val, 2))


class IlluminationSmoothnessLoss(nn.Module):
    """Smoothness regularizer on the predicted curve parameter maps."""
    def forward(self, x):
        batch_size, _, h, w = x.shape
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (h_tv / (h - 1) + w_tv / (w - 1)) / (batch_size * h * w)


class SpatialConsistencyLoss(nn.Module):
    """Preserves spatial coherence between input and enhanced image via local region differences."""
    def __init__(self):
        super().__init__()
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)
        self.pool = nn.AvgPool2d(4)

    def _grad(self, x, weight):
        return F.conv2d(x, weight, padding=1)

    def forward(self, org, enhance):
        org_mean = org.mean(dim=1, keepdim=True)
        enhance_mean = enhance.mean(dim=1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left = self._grad(org_pool, self.weight_left)
        d_org_right = self._grad(org_pool, self.weight_right)
        d_org_up = self._grad(org_pool, self.weight_up)
        d_org_down = self._grad(org_pool, self.weight_down)

        d_enh_left = self._grad(enhance_pool, self.weight_left)
        d_enh_right = self._grad(enhance_pool, self.weight_right)
        d_enh_up = self._grad(enhance_pool, self.weight_up)
        d_enh_down = self._grad(enhance_pool, self.weight_down)

        d_left = torch.pow(d_org_left - d_enh_left, 2)
        d_right = torch.pow(d_org_right - d_enh_right, 2)
        d_up = torch.pow(d_org_up - d_enh_up, 2)
        d_down = torch.pow(d_org_down - d_enh_down, 2)
        return torch.mean(d_left + d_right + d_up + d_down)


def zero_dce_loss(low_light_img, enhanced_img, curve_maps,
                   w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0):
    """Combined unsupervised loss used to train Zero-DCE."""
    spa_loss = SpatialConsistencyLoss().to(low_light_img.device)
    exp_loss = ExposureLoss().to(low_light_img.device)
    col_loss = ColorConstancyLoss().to(low_light_img.device)
    tv_loss = IlluminationSmoothnessLoss().to(low_light_img.device)

    l_spa = w_spa * spa_loss(low_light_img, enhanced_img)
    l_exp = w_exp * exp_loss(enhanced_img)
    l_col = w_col * col_loss(enhanced_img)
    l_tv = w_tv * tv_loss(curve_maps)

    total = l_spa + l_exp + l_col + l_tv
    return total, {"spa": l_spa.item(), "exp": l_exp.item(), "col": l_col.item(), "tv": l_tv.item()}
