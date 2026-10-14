import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DeformConv3d(nn.Module):
    """
    Deformable Conv 3D (DCNv1, thuần PyTorch, dùng grid_sample).
    - 1 offset group dùng chung cho mọi kênh (81 = 27 tap x 3 trục)
    - Offset: DW 3x3x3 -> PW 1x1x1, PW được zero-init => ban đầu = Conv3d thường
    - Hỗ trợ groups cho phần weight
    """
    def __init__(self, in_ch, out_ch, groups=1, use_checkpoint=False):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0
        self.in_ch, self.out_ch, self.groups = in_ch, out_ch, groups
        self.use_checkpoint = use_checkpoint
        self.K = 27

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, self.K))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight.view(out_ch, -1), a=5 ** 0.5)

        # nhánh dự đoán offset (nhẹ)
        self.off_dw = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.off_pw = nn.Conv3d(in_ch, self.K * 3, 1)
        nn.init.zeros_(self.off_pw.weight)   # zero-init
        nn.init.zeros_(self.off_pw.bias)

        # toạ độ tap (dz, dy, dx) theo thứ tự z, y, x
        taps = torch.stack(torch.meshgrid(
            *[torch.tensor([-1., 0., 1.])] * 3, indexing="ij"), dim=-1).view(-1, 3)
        self.register_buffer("taps_zyx", taps, persistent=False)

    def _sample_tap(self, x, k, off_xyz, base, scale):
        # off_xyz: (B, 27, 3, D, H, W) theo thứ tự x, y, z (đơn vị voxel)
        tap_xyz = self.taps_zyx[k].flip(0)                     # (3,) x,y,z
        o = off_xyz[:, k].permute(0, 2, 3, 4, 1)               # (B, D, H, W, 3)
        grid = base + (o + tap_xyz) * scale
        s = F.grid_sample(x, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=True)
        w = self.weight[:, :, k].reshape(self.out_ch, self.in_ch // self.groups, 1, 1, 1)
        return F.conv3d(s, w, groups=self.groups)

    def forward(self, x):
        B, C, D, H, W = x.shape
        off = self.off_pw(self.off_dw(x)).view(B, self.K, 3, D, H, W)
        off_xyz = off.flip(2)                                  # (z,y,x) -> (x,y,z)

        # lưới cơ sở chuẩn hoá [-1, 1], thứ tự (x, y, z)
        zs = torch.linspace(-1, 1, D, device=x.device, dtype=x.dtype)
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        base = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)  # (1, D, H, W, 3)
        scale = torch.tensor([2. / max(W - 1, 1), 2. / max(H - 1, 1), 2. / max(D - 1, 1)],
                             device=x.device, dtype=x.dtype)

        out = 0
        for k in range(self.K):
            if self.use_checkpoint and self.training:
                out = out + checkpoint(self._sample_tap, x, k, off_xyz, base, scale,
                                       use_reentrant=False)
            else:
                out = out + self._sample_tap(x, k, off_xyz, base, scale)
        return out + self.bias.view(1, -1, 1, 1, 1)


class CoordAttention3D(nn.Module):
    """3D Coordinate Attention (dùng GroupNorm thay BatchNorm)."""
    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.reduce = nn.Conv3d(ch, mid, 1)
        self.gn = nn.GroupNorm(4 if mid % 4 == 0 else 1, mid)
        self.act = nn.Hardswish()
        self.att_d = nn.Conv3d(mid, ch, 1)
        self.att_h = nn.Conv3d(mid, ch, 1)
        self.att_w = nn.Conv3d(mid, ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        pd = x.mean(dim=(3, 4), keepdim=True)                              # (B,C,D,1,1)
        ph = x.mean(dim=(2, 4), keepdim=True).permute(0, 1, 3, 2, 4)       # (B,C,H,1,1)
        pw = x.mean(dim=(2, 3), keepdim=True).permute(0, 1, 4, 2, 3)       # (B,C,W,1,1)

        y = self.act(self.gn(self.reduce(torch.cat([pd, ph, pw], dim=2))))
        yd, yh, yw = torch.split(y, [D, H, W], dim=2)

        ad = torch.sigmoid(self.att_d(yd))                                  # (B,C,D,1,1)
        ah = torch.sigmoid(self.att_h(yh)).permute(0, 1, 3, 2, 4)           # (B,C,1,H,1)
        aw = torch.sigmoid(self.att_w(yw)).permute(0, 1, 3, 4, 2)           # (B,C,1,1,W)
        return x * ad * ah * aw


class ShiftDepthwise(nn.Module):
    """Một offset 3D dùng chung cho mọi kênh, rồi depthwise 3x3x3.

    Tên tham số khớp checkpoint đã train: off_dw, off_pw (3 kênh), dw.
    """
    def __init__(self, ch):
        super().__init__()
        self.off_dw = nn.Conv3d(ch, ch, 3, padding=1, groups=ch)
        self.off_pw = nn.Conv3d(ch, 3, 1)
        self.dw = nn.Conv3d(ch, ch, 3, padding=1, groups=ch)
        nn.init.zeros_(self.off_pw.weight)
        nn.init.zeros_(self.off_pw.bias)

    def forward(self, x):
        batches, _, depth, height, width = x.shape
        offset = self.off_pw(self.off_dw(x)).permute(0, 2, 3, 4, 1)
        zs = torch.linspace(-1, 1, depth, device=x.device, dtype=x.dtype)
        ys = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        base = torch.stack([gx, gy, gz], dim=-1)
        scale = torch.tensor(
            [2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1), 2.0 / max(depth - 1, 1)],
            device=x.device,
            dtype=x.dtype,
        )
        grid = (base + offset.flip(-1) * scale).expand(batches, -1, -1, -1, -1)
        warped = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return self.dw(warped)


class DeformStem(nn.Module):
    def __init__(self, in_ch=1, C=32, gn_groups=8, mid=16, use_checkpoint=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, mid, 3, stride=1, padding=1)
        self.act = nn.GELU()
        self.deform = ShiftDepthwise(mid)
        self.proj = nn.Conv3d(mid, C, 1)
        self.gn = nn.GroupNorm(gn_groups, C)
        self.ca = CoordAttention3D(C)
        self.res = nn.Conv3d(in_ch, C, 1)

    def forward(self, x):
        hidden = self.deform(self.act(self.conv1(x)))
        hidden = self.ca(self.gn(self.act(self.proj(hidden))))
        return hidden + self.res(x)
