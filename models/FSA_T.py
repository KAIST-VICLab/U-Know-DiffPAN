import math
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from timm.models.layers import DropPath

class UNetSR3(nn.Module):
    def __init__(
        self,
        in_channel=8,
        out_channel=3,
        inner_channel=32,
        lms_channel=8,
        pan_channel=1,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        self_condition=False,
        fourier_features=False,
        fourier_min=7,
        fourier_max=8,
        fourier_step=1,
        pred_var=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.lms_channel = lms_channel
        self.pan_channel = pan_channel

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        if self_condition:
            in_channel += out_channel
        if fourier_features:
            n = np.ceil((fourier_max - fourier_min) / fourier_step).astype("int")
            in_channel += in_channel * n * 2

        self.fourier_features = fourier_features
        self.fourier_min = fourier_min
        self.fourier_max = fourier_max
        self.fourier_step = fourier_step

        self.pred_var = pred_var

        cond_dim =  {
                    'lms': lms_channel, 'pan': pan_channel,
                    'pre_out': lms_channel, 'pre_variance': lms_channel, 
                    'wavelets': lms_channel + 3*pan_channel, 'pre_wavelets': 4*lms_channel
                    }
        
        # prior cond extractor
        self.prior_cond_extractor = PriorCondExtractor(in_channels=lms_channel, n_feats=inner_channel)
        downs = [nn.Conv2d(in_channel+lms_channel+pan_channel, inner_channel, kernel_size=3, padding=1)]                
        
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        cond_dim = cond_dim,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        encoder=True,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult

            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            if use_attn:
                print("use attn: res {}".format(now_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                        encoder=False,
                        cond_dim = cond_dim,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(
            # confidence map channel 
            pre_channel, default(out_channel+out_channel, in_channel), groups=norm_groups
        )

        self.res_blocks = res_blocks
        self.self_condition = self_condition

    def forward(self, x, time, cond=None, self_cond=None):
        # self-conditioning
        prior_z = self.prior_cond_extractor(cond['pre_out'], cond['pre_variance'])
        # cond dict 에 prior_z 추가
        cond['prior_z'] = prior_z
        if self.self_condition:
            self_cond = default(self_cond, x)
            
            in_cond = torch.cat([cond['lms'], cond['pan']], dim=1)
            x = torch.cat([self_cond, in_cond, x], dim=1)

        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []
        dist_feature_map = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    x, 
                    t, 
                    cond
                )  # cond: cat[lms, pan]
            else:
                if not self.training:
                    dist_feature_map.append(x)
                x = layer(x)
            
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                if not self.training:
                    dist_feature_map.append(x)
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    torch.cat((x, feats.pop()), dim=1),
                    t,
                    cond,
                )  # cond: cat[lms_main, pan_h, pan_v]
            else:
                if not self.training:
                    dist_feature_map.append(x)
                x = layer(x)
        
        x = self.final_conv(x)

        # Assuming x has shape [batch_size, channels, height, width]
        variance = F.softplus(x[:, -self.out_channel:, :, :])
        out = x[:, :-self.out_channel, :, :]
        return {'out' : out, 'variance' : variance, 'dist_feature_map' : dist_feature_map}  # + res

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseModulation(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseModulation, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )
    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )
    def forward(self, x):
        return self.block(x)

# -------------ResNet Block (With time embedding)----------------------------------------
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseModulation(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb):
        # b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

# -------------ResNet Block (No time embedding)----------------------------------------
class Resblock(nn.Module):
    def __init__(self,in_channels=64,out_channels=64):
        super(Resblock, self).__init__()

        channel = 64
        self.conv21 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv22 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv21(x))  # Bsx64x64x64
        rs1 = self.relu(self.conv22(rs1))   # Bsx64x64x64
        rs = torch.add(x, rs1)  # Bsx64x64x64
        return rs

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input

class SelfChannelAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        query, key, value = map(
            lambda in_x: rearrange(
                in_x, "b h c xf yf -> b h c (xf yf)", h=n_head
            ),
            (query, key, value),
        )
        # c x c attn map
        attn = torch.einsum("b h j n, b h k n -> b h j k", query, key).contiguous() / math.sqrt(channel)
        attn = attn.softmax(-1)
        # h w fused feature map
        out = torch.einsum("b h j k, b h k n-> b h j n", attn, value)
        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=height, yf=width, h=n_head
        )

        return out + input

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

class PriorCondInjection(nn.Module):
    def __init__(self, fea_dim, hidden_dim) -> None:
        super().__init__()
        
        self.kernel = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, fea_dim * 2, bias=False),
        )
        self.x_conv = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim*2, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim*2, fea_dim, 1, bias=True),
        )
        self.x_attn = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim*2, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim*2, fea_dim, 1, bias=True),
        )
        self.project_out = nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, bias=True)
        
        nn.init.zeros_(self.kernel[-1].weight)
        nn.init.zeros_(self.x_conv[-1].weight)
        nn.init.zeros_(self.x_conv[-1].bias)
        nn.init.zeros_(self.x_attn[-1].weight)
        nn.init.zeros_(self.x_attn[-1].bias)

    def forward(self, x, prior_z):

        prior_z = self.kernel(prior_z)
        shift, scale = prior_z.chunk(2, dim=1)
        b, c = shift.shape
        
        res = x
        x = x * (1 + scale.view(b,c,1,1)) + shift.view(b,c,1,1)
        x_feat = self.x_conv(x)
        x_attn = F.gelu(self.x_attn(x))
        x = x_feat * x_attn + res
        x = self.project_out(x)
        return x

class PriorCondExtractor(nn.Module):
    def __init__(self, in_channels = 8,  n_feats = 64, n_encoder_res = 3):
        super().__init__()
        E1=[nn.Conv2d(2*in_channels, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            Resblock(in_channels=n_feats,
                     out_channels=n_feats) for _ in range(n_encoder_res)
        ]

        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),      # 64 128
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),  # 128 128
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats *2 , kernel_size=3, padding=1),  # 128 64
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E_12 = E1 + E2
        self.E_12 = nn.Sequential(
            *E_12
        )
        self.E3 = nn.Sequential(
            *E3
        ) 
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_feats *2, 128),
        )

    def forward(self, pre_out, pre_variance):
        x = torch.cat([pre_out, pre_variance], dim=1)   # 6 256 256
        mid = self.E_12(x)
        # channel z
        fea1 = self.E3(mid).squeeze(-1).squeeze(-1)
        prior_z = self.mlp(fea1)
        return prior_z  # output size : 1 * 128


class FreqCondInjection(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)

        self.q = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim, 1, bias=True),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, 1, 1, bias=False, groups=cond_dim),
            nn.Conv2d(cond_dim, qkv_dim * 2, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, cond):
        x = self.prenorm_x(x)

        q = self.q(x)
        k, v = self.kv(cond).chunk(2, dim=1)

        q, k, v = map(lambda in_qkv: F.normalize(in_qkv, dim=1), (q, k, v))

        # convert to freq space
        q = torch.fft.rfft2(q, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        k = torch.fft.rfft2(k, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        v = torch.fft.rfft2(v, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1

        # amp and phas attention
        amp_out = self.attn_op(q.abs(), k.abs(), v.abs())
        phas_out = self.attn_op(q.angle(), k.angle(), v.angle())

        # convert to complex
        out = torch.polar(amp_out, phas_out)

        # convert to rgb space
        out = torch.fft.irfft2(out, dim=(-2, -1), norm="ortho")

        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out

    def attn_op(self, q, k, v):
        b, c, xf, yf = q.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        # n x n attn map
        # c x c attn map
        sim = torch.einsum("b h cx hw, b h cy hw -> b h cx cy", q, k) * self.scale
        sim = sim.softmax(-1)
        # h w fused feature map
        out = torch.einsum("b h cx cy, b h cy hw-> b h cx hw", sim, v)
        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )
        return out

class FreqCA(nn.Module):
    def __init__(
        self,
        fea_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0, "@dim must be divisible by @nheads"

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)

        self.qkv = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim * 3, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x):
        x = self.prenorm_x(x)

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda in_qkv: F.normalize(in_qkv, dim=1), (q, k, v))

        # convert to freq space
        q = torch.fft.rfft2(q, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        k = torch.fft.rfft2(k, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        v = torch.fft.rfft2(v, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1

        # amp and phas attention
        amp_out = self.attn_op(q.abs(), k.abs(), v.abs())
        phas_out = self.attn_op(q.angle(), k.angle(), v.angle())

        # convert to complex
        out = torch.polar(amp_out, phas_out)

        # convert to rgb space
        out = torch.fft.irfft2(out, dim=(-2, -1), norm="ortho")

        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out

    def attn_op(self, q, k, v):
        b, c, xf, yf = q.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        # n x n attn map
        # c x c attn map
        sim = torch.einsum("b h j n, b h k n -> b h j k", q, k) * self.scale
        sim = sim.softmax(-1)
        # h w fused feature map
        out = torch.einsum("b h j k, b h k n-> b h j n", sim, v)
        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )
        return out

class FastAttnCondInjection_qk_v(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0, "@dim must be divisible by @nheads"

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)
  
        self.qk = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, 1, 1, bias=False, groups=cond_dim),
            nn.Conv2d(cond_dim, qkv_dim * 2, 1, bias=True),
        )
        self.v = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, cond):
        x = self.prenorm_x(x)

        q, k = self.qk(cond).chunk(2, dim=1)
        v = self.v(x)

        k = k.softmax(dim=-2)
        v = v.softmax(dim=-1)

        b, c, xf, yf = v.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        v = v *self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", q, k)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, v)

        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )

        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out
    
class FastAttnCondInjection_q_kv(nn.Module):
    def __init__(
        self,
        fea_dim,
        cond_dim,
        qkv_dim,
        dim_out,
        groups=32,
        nheads=8,
        drop_path_prob=0.2,
    ) -> None:
        super().__init__()
        assert fea_dim % nheads == 0, "@dim must be divisible by @nheads"

        self.prenorm_x = nn.GroupNorm(groups, fea_dim)

        self.kv = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, 1, 1, bias=False, groups=cond_dim),
            nn.Conv2d(cond_dim, qkv_dim * 2, 1, bias=True),
        )
        self.q = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, 3, 1, 1, bias=False, groups=fea_dim),
            nn.Conv2d(fea_dim, qkv_dim, 1, bias=True),
        )
        self.nheads = nheads
        self.scale = 1 / math.sqrt(qkv_dim // nheads)

        self.attn_out = nn.Conv2d(qkv_dim, dim_out, 1, bias=True)
        self.attn_res = (
            nn.Conv2d(fea_dim, dim_out, 1, bias=True)
            if fea_dim != dim_out
            else nn.Identity()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(dim_out, dim_out * 2, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, 1, bias=False),
            nn.Conv2d(dim_out, dim_out, 1, bias=True),
        )
        self.ffn_drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, cond):
        x = self.prenorm_x(x)
 
        k, v = self.kv(cond).chunk(2, dim=1)
        q = self.q(x)

        k = k.softmax(dim=-2)
        v = v.softmax(dim=-1)

        b, c, xf, yf = v.shape

        q, k, v = map(
            lambda in_x: rearrange(
                in_x, "b (h c) xf yf -> b h c (xf yf)", h=self.nheads
            ),
            (q, k, v),
        )
        v = v *self.scale

        # c x c attn map

        context = torch.einsum("b h d n, b h e n -> b h d e", q, k)

        # h w fused feature map
        out = torch.einsum("b h d e, b h d n -> b h e n", context, v)

        out = rearrange(
            out, "n h c (xf yf) -> n (h c) xf yf", xf=xf, yf=yf, h=self.nheads
        )

        # convert to rgb space
        attn_out = self.attn_out(out) + self.attn_res(x)

        # ffn
        ffn_out = self.ffn_drop_path(self.ffn(attn_out)) + attn_out
        return ffn_out

class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        dwt_cond_dim=None,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        encoder=True,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.encoder = encoder
        self.cond_dim = cond_dim # dict('lms', 'pan', 'pre_out', 'pre_variance', 'wavelets', 'pre_wavelets')
        self.dwt_cond_dim = dwt_cond_dim
        self.with_cond = exists(cond_dim)
        self.res_block = ResnetBlock(
            dim_out if exists(cond_dim) else dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        if self.with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups, n_head=8)
        if self.with_cond:
            if encoder:
                self.prior_cond_inj = PriorCondInjection(
                    dim, dim_out    # pre_out + lms + pan
                )

            else:
                if self.with_attn:
                    self.dec_res_block = ResnetBlock(
                        dim,
                        dim_out,
                        noise_level_emb_dim,
                        norm_groups=norm_groups,
                        dropout=dropout,
                    )
                else:
                    self.attn_res = (
                        nn.Conv2d(dim, dim_out, 1, bias=True)
                        if dim != dim_out
                        else nn.Identity()
                    )
                    self.time_emd_block = ResnetBlock(
                        dim,
                        dim,
                        noise_level_emb_dim,
                        norm_groups=norm_groups,
                        dropout=dropout,
                    )
                    self.fourier_CA = FreqCA(
                        dim,
                        dim,
                        dim,
                        groups = norm_groups,
                        nheads= 8,
                        drop_path_prob=0.2
                    )
                    self.cond_inj_qk_v = FastAttnCondInjection_qk_v(
                        dim,
                        cond_dim['wavelets'],    # wavelets + pre_variance
                        dim,
                        dim,
                        groups=norm_groups,
                        nheads=8,
                        drop_path_prob=0.2,
                    )
                    self.cond_inj_q_kv = FastAttnCondInjection_q_kv(
                        dim,
                        cond_dim['wavelets'],   # wavelets
                        dim,
                        dim_out,
                        groups=norm_groups,
                        nheads=8,
                        drop_path_prob=0.2,
                    )

    def forward(self, x, time_emb, cond=None):
        # condition injection
        if self.with_cond:
            wavelets_cond = F.interpolate(cond['wavelets'], size=x.shape[-2:], mode="bilinear")
            if self.encoder:
                prior_z = cond['prior_z']
                x = self.prior_cond_inj(x, prior_z)
            else:
                if self.with_attn:
                    x = self.dec_res_block(x, time_emb)
                else:
                    res = self.attn_res(x)
                    x = self.time_emd_block(x, time_emb)
                    x = self.fourier_CA(x)
                    x = self.cond_inj_qk_v(
                        x, wavelets_cond
                    )
                    x = self.cond_inj_q_kv(
                        x, wavelets_cond
                    )
                    x = x + res

        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

if __name__ == "__main__":

    # context to test runtime
    import contextlib
    import time
    
    @contextlib.contextmanager
    def time_it(t=10):
        t1 = time.time()
        yield
        t2 = time.time()
        print('total time: {}, ave time: {:.3f}s'.format((t2-t1), (t2 - t1)/t))
        
    device = 'cuda:1'

    net = UNetSR3(
        in_channel=8,
        channel_mults=(1, 2, 2, 4),
        out_channel=8,
        lms_channel=8,
        pan_channel=1,
        image_size=64,
        self_condition=True,
        inner_channel=32,
        norm_groups=1,
        attn_res=(8,),
        dropout=0.2,
    ).to(device)
    x = torch.randn(1, 31, 512, 512).to(device)
    t = torch.LongTensor([1]).to(device)
    
    lms = torch.randn(1, 8, 512, 512).to(device)
    pan = torch.randn(1, 1, 512, 512).to(device)
    pre_out = torch.randn(1, 8, 512, 512).to(device)
    pre_variance = torch.randn(1, 8, 512, 512).to(device)
    wavelets = torch.randn(1, 11, 512, 512).to(device)
    cond =  {
        'lms': lms, 'pan': pan,
        'pre_out': pre_out, 'pre_variance': pre_variance, 
        'wavelets': wavelets, 
        }
    self_cond = torch.randn(1, 8, 512, 512).to(device)

    with torch.no_grad():
        y = net(x, t, cond, self_cond)
        tt = 25
        with time_it(tt):
            for _ in range(tt):
                y = net(x, t, cond)
    
