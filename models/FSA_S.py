import math
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from timm.models.layers import DropPath

# from diffusion_engine import norm

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
        res_blocks=2,       
        dropout=0.2,
        with_noise_level_emb=True,
        image_size=128,
        self_condition=False,
    ):
        super().__init__()

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

        cond_dim =  {
                    'lms': lms_channel, 'pan': pan_channel,
                    'pre_out': lms_channel, 'pre_variance': lms_channel, 
                    'wavelets': lms_channel + 3*pan_channel, 'pre_wavelets': 4*lms_channel
                    }

        # SR3 할때는 input 에 pan + lms 추가 해줘야함
        downs = [nn.Conv2d(in_channel+lms_channel+pan_channel, inner_channel, kernel_size=3, padding=1)]
        
        # SR3 아닌경우는 어떨지 모르겠넹 일단 추가 안하는쪽으로 학습했으니 안하는걸로
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        cond_dim=lms_channel + pan_channel,
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
                        cond_dim=cond_dim,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )

        self.res_blocks = res_blocks
        # X_t+1
        self.self_condition = self_condition

    def forward(self, x, time, cond=None, self_cond = None):
        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, x)

            in_cond = torch.cat([cond['lms'], cond['pan']], dim=1)
            x = torch.cat([self_cond, in_cond, x], dim=1)

        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []
        dist_feature_map = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)  # cond: cat[lms, pan]
            else:
                dist_feature_map.append(x)
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                dist_feature_map.append(x)
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    torch.cat((x, feats.pop()), dim=1),
                    t
                )
            else:
                dist_feature_map.append(x)
                x = layer(x)
        out = self.final_conv(x)
        return {'out' : out, 'dist_feature_map' : dist_feature_map}  # + res

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

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
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
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

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

class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        encoder=True,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.encoder = encoder

        self.res_block = ResnetBlock(
            dim_out if exists(cond_dim) else dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups, n_head=8)
        if encoder:
            self.enc_res_block =ResnetBlock(
                    dim,
                    dim_out,
                    noise_level_emb_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
        else:
            if self.with_attn:
                self.mid_res_block = ResnetBlock(
                    dim,
                    dim_out,
                    noise_level_emb_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            else:
                self.time_emd_block = ResnetBlock(
                    dim,
                    dim,
                    noise_level_emb_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
                self.dec_res_block = ResnetBlock(
                    dim,
                    dim_out,
                    noise_level_emb_dim,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
 
    def forward(self, x, time_emb):
        if self.encoder:
            x = self.enc_res_block(x, time_emb)
        else:
            if self.with_attn:
                x = self.mid_res_block(x, time_emb)
            else:
                x = self.time_emd_block(x, time_emb)
                x = self.dec_res_block(x, time_emb)
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
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    
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
        in_channel=31,
        channel_mults=(1, 2, 2, 2),
        out_channel=8,
        lms_channel=8,
        pan_channel=1,
        image_size=64,
        self_condition=False,
        inner_channel=32,
        norm_groups=1,
        attn_res=(8,),
        dropout=0.2,
    ).to(device)
    x = torch.randn(1, 31, 512, 512).to(device)
    cond = torch.randn(1, 31 + 1 + 31 + 3, 512, 512).to(device)  # [lms, pan, lms_main, h, v, d]
    t = torch.LongTensor([1]).to(device)
    
    with torch.no_grad():
        y = net(x, t, cond)
        tt = 25
        with time_it(tt):
            for _ in range(tt):
                y = net(x, t, cond)
 
    
