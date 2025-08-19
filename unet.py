import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass, asdict
from einops import rearrange
from collections import deque


@dataclass
class UNetConfig:
    in_channels: int
    base_hidden: int
    scales: list[int]
    n_heads: int
    p_drop: float
    attn_resolutions: list[int]
    in_resolution: int
    pred_variance: bool
    n_resblocks: int

    def as_dict(self):
        return asdict(self)


class TimeEmbed(nn.Module):
    def __init__(self, embed_channels: int, channels: int):
        super().__init__()

        self.mlp = nn.Linear(embed_channels, channels)

    def forward(self, t: torch.Tensor):
        return self.mlp(t)[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_chann: int,
        out_chann: int,
        p_drop: float = 0.0,
    ):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_chann, out_chann, (3, 3), stride=1, padding=1)
        self.conv_2 = nn.Conv2d(out_chann, out_chann, (3, 3), stride=1, padding=1)

        self.norm1 = nn.GroupNorm(32, in_chann)
        self.norm2 = nn.GroupNorm(32, out_chann)

        self.t_embed_wb = TimeEmbed(4 * config.base_hidden, 2 * out_chann)

        self.skip_connection = nn.Identity()
        if in_chann != out_chann:
            self.skip_connection = nn.Conv2d(in_chann, out_chann, (1, 1))

        self.dropout = nn.Dropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv_2.weight)

        assert self.conv_2.bias is not None
        torch.nn.init.zeros_(self.conv_2.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        resid = x

        x = self.conv_1(F.silu(self.norm1(x)))

        wb = self.t_embed_wb(t)

        w, b = torch.chunk(wb, chunks=2, dim=1)

        x = (w + 1) * self.norm2(x) + b
        x = self.conv_2(self.dropout(F.silu(x)))

        return (self.skip_connection(resid) + x) / math.sqrt(2.0)


class CHWAttention(nn.Module):
    def __init__(self, config: UNetConfig, hidden_size: int):
        super().__init__()
        self.config = config

        self.nh = config.n_heads

        self.qkv = nn.Linear(
            in_features=hidden_size,
            out_features=3 * hidden_size,
        )
        self.out = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        _, _, h, w = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")

        x = self.norm(x)

        qkv = self.qkv(x)

        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        queries = rearrange(q, "b l (nh hd) -> b nh l hd", nh=self.nh)
        keys = rearrange(k, "b l (nh hd) -> b nh l hd", nh=self.nh)
        values = rearrange(v, "b l (nh hd) -> b nh l hd", nh=self.nh)

        attn_out = F.scaled_dot_product_attention(queries, keys, values)

        o = self.out(rearrange(attn_out, "b nh l hd -> b l (nh hd)"))

        out: torch.Tensor = rearrange(o, "b (h w) c -> b c h w", h=h, w=w) + resid

        out = out.contiguous()

        return out / math.sqrt(2.0)


class Stage(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        attention: bool = False,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        self.blocks.append(
            ResidualBlock(config, in_channels, hidden_channels, p_drop=config.p_drop)
        )
        self.attns.append(
            CHWAttention(config, hidden_channels) if attention else nn.Identity()
        )

        for _ in range(num_blocks - 2):
            self.blocks.append(
                ResidualBlock(
                    config, hidden_channels, hidden_channels, p_drop=config.p_drop
                )
            )
            self.attns.append(
                CHWAttention(config, hidden_channels) if attention else nn.Identity()
            )

        self.blocks.append(
            ResidualBlock(config, hidden_channels, out_channels, p_drop=config.p_drop)
        )
        self.attns.append(
            CHWAttention(config, out_channels) if attention else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, t)
            x = attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()

        self.config = config

        self.pred_variance = self.config.pred_variance

        self.base_channs = config.base_hidden

        channs_1 = config.scales[0] * self.base_channs

        self.conv_in = nn.Conv2d(
            config.in_channels, channs_1, kernel_size=(3, 3), padding=1, bias=True
        )

        down_stages = deque()
        up_stages = deque()

        # build the UNet stages
        res = config.in_resolution
        for i in range(0, len(config.scales) - 1):
            scales_curr, scales_next = config.scales[i], config.scales[i + 1]

            chann_curr, chann_next = (
                self.base_channs * scales_curr,
                self.base_channs * scales_next,
            )

            do_attn = res in config.attn_resolutions

            stage_down = Stage(
                config,
                chann_curr,
                chann_curr,
                chann_next,
                attention=do_attn,
                num_blocks=config.n_resblocks,
            )

            stage_up = Stage(
                config,
                chann_next + chann_next,
                chann_curr,
                chann_curr,
                attention=do_attn,
                num_blocks=config.n_resblocks + 1,
            )

            down_stages.append(stage_down)
            up_stages.appendleft(stage_up)

            res = res // 2

        mid_stage = config.scales[-1] * self.base_channs

        stage_down = Stage(
            config, mid_stage, mid_stage, mid_stage, num_blocks=config.n_resblocks
        )
        stage_up = Stage(
            config,
            mid_stage + mid_stage,
            mid_stage,
            mid_stage,
            num_blocks=config.n_resblocks + 1,
        )

        down_stages.append(stage_down)
        up_stages.appendleft(stage_up)

        self.mid = Stage(
            config,
            mid_stage,
            mid_stage,
            mid_stage,
            attention=True,
            num_blocks=config.n_resblocks,
        )

        self.down_stages = nn.ModuleList(list(down_stages))
        self.up_stages = nn.ModuleList(list(up_stages))

        out_channels = (
            self.config.in_channels * 2
            if self.pred_variance
            else self.config.in_channels
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, channs_1),
            nn.SiLU(),
            nn.Conv2d(channs_1, out_channels, kernel_size=(3, 3), padding=1, bias=True),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channs, 4 * self.config.base_hidden),
            nn.SiLU(),
            nn.Linear(4 * self.config.base_hidden, 4 * self.config.base_hidden),
            nn.SiLU(),
        )

        self.reset_parameters()

        self.downsample = partial(F.interpolate, scale_factor=0.5, mode="nearest")
        self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")

    def _timestep(self, t: torch.Tensor):
        half = self.base_channs // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (math.log(10000.0) / (half))
        )
        emb = (t)[:, None].float() * freqs[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.base_channs % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]

    def reset_parameters(self):
        assert isinstance(self.out[-1], nn.Conv2d)
        torch.nn.init.zeros_(self.out[-1].weight)  # type: ignore

        if self.out[-1].bias is not None:
            torch.nn.init.zeros_(self.out[-1].bias)  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_embed(self._timestep(t))

        act = self.conv_in(x)  # [B,C,H,W]

        hs = []

        for i, stage in enumerate(self.down_stages):
            if i > 0:
                act = stage(self.downsample(act), t_emb)
            else:
                act = stage(act, t_emb)

            hs.append(act)

        act = self.mid(act, t_emb)

        for i, stage in enumerate(self.up_stages):
            h = hs.pop()
            if i > 0:
                act = stage(torch.concat([self.upsample(act), h], dim=1), t_emb)
            else:
                act = stage(torch.concat([act, h], dim=1), t_emb)

        out = self.out(act)  # [B,C,H,W] | [B,2*C,H,W]

        if self.pred_variance:
            pred_eps, pred_v = torch.chunk(out, chunks=2, dim=1)
            return pred_eps, (pred_v + 1.0) / 2.0

        else:
            return out
