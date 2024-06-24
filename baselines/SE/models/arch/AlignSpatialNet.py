from typing import *

import torch
import torch.nn as nn
from models.arch.base.norm import *
from models.arch.base.non_linear import *
from models.arch.base.linear_group import LinearGroup
from torch import Tensor
from torch.nn import MultiheadAttention
import torch.utils.checkpoint as cp


class CrossAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, merge: Literal['add', 'cat_add', 'cat']) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)])

        self.merge_type = merge
        self.merge = None if merge == 'add' else nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        x0 = x
        x, k, v = [self.norms[i](j) for i, j in enumerate([x, k, v])]
        o = self.mha.forward(x, k, v, need_weights=False, attn_mask=attn_mask)
        if self.merge is not None:
            return self.merge(torch.concat([x0, o], dim=-1))
        else:
            return o


class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
            cfg: List[str] = [],
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        self.cfg = cfg
        # MHSA module
        if '-mhsa' not in cfg:
            self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
            self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
            self.dropout_mhsa = nn.Dropout(dropout[0])

        # Cross-attention Module
        if '+mhca' in cfg:
            self.norm_mhca = nn.ModuleList([new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups) for i in range(2)])
            self.mhca = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
            self.dropout_mhca = nn.Dropout(dropout[0])

        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    # def forward(self, x: Tensor, kv: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
    #     r"""
    #     Args:
    #         x: shape [B, F, T, H]
    #         att_mask: the mask for attention along T. shape [B, T, T]

    #     Shape:
    #         out: shape [B, F, T, H]
    #     """
    #     # raise NotImplementedError('not implemented')
    #     # x = x + self._fconv(self.fconv1, x)
    #     # x = x + self._full(x)
    #     # x = x + self._fconv(self.fconv2, x)
    #     # if '-mhsa' not in self.cfg:
    #     #     x_, attn = self._tsa(x, att_mask, kv=kv)
    #     #     x = x + x_
    #     # else:
    #     #     attn = None
    #     # x = x + self._tconvffn(x)
    #     return x

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        x, attn = self.mhsa.forward(x, x, x, need_weights=False, average_attn_weights=False, attn_mask=attn_mask)
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tca(self, x: Tensor, kv: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        assert kv is not None
        x, kv = self.norm_mhca[0](x), self.norm_mhca[1](kv)
        x, kv = x.reshape(B * F, T, H), kv.reshape(B * F, kv.shape[2], H)
        x, attn = self.mhca.forward(x, kv, kv, need_weights=False, average_attn_weights=False, attn_mask=attn_mask)
        x = x.reshape(B, F, T, H)
        return self.dropout_mhca(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


def get_local_mask(Tq: int, Tkv: int, allow_time_dura: float, device):
    pos1 = torch.arange(start=0, end=Tq, dtype=torch.float32, device=device, requires_grad=False).unsqueeze(1) / Tq
    pos2 = torch.arange(start=0, end=Tkv, dtype=torch.float32, device=device, requires_grad=False).unsqueeze(0) / Tkv
    relative_pos = torch.abs(pos1 - pos2)
    mask = torch.where(relative_pos <= allow_time_dura, 0, -torch.inf)
    return mask


class AlignSpatialNet(nn.Module):

    def __init__(
            self,
            dim_input: int = 2,  # the input dim for each time-frequency point
            dim_output: int = 2,  # the output dim for each time-frequency point
            dim_squeeze: int = 8,
            num_layers: int = 8,
            num_freqs: int = 129,
            dim_hidden: int = 96,
            dim_ffn: int = 192,
            cross_mask: int = 13,  # 200ms for left & 200ms for right
            cfg_cln: List[str] = ['-mhsa', '+mhca'],
            cfg_mix: List[str] = ['+mhca'],
            output: Literal['mix', 'cln', 'mix+cln'] = 'mix',
            activation_checkpointing: bool = False,  # saving lots of memories when the model is large
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=5, stride=1, padding="same"),
            nn.Conv1d(in_channels=2, out_channels=dim_hidden, kernel_size=5, stride=1, padding="same"),
        ])

        self.sp_cln = nn.ModuleList([SpatialNetLayer(
            dim_hidden=dim_hidden,
            dim_ffn=dim_ffn,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            num_heads=4,
            cfg=cfg_cln,
        ) for l in range(num_layers)])

        self.sp_mix = nn.ModuleList([SpatialNetLayer(
            dim_hidden=dim_hidden,
            dim_ffn=dim_ffn,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            num_heads=4,
            cfg=cfg_mix,
        ) for l in range(num_layers)])

        self.decoder = nn.Linear(in_features=dim_hidden * (2 if '+' in output else 1), out_features=dim_output)
        self.cross_mask_len = cross_mask
        self.num_layers = num_layers

        self.cfg_cln = cfg_cln
        self.cfg_mix = cfg_mix
        self.output = output
        self.activation_checkpointing = activation_checkpointing

    def forward(self, mix: Tensor, cln: Tensor) -> Tensor:
        B, F, T, H0 = mix.shape
        mix = self.encoder[0](mix.reshape(B * F, mix.shape[2], H0).permute(0, 2, 1)).permute(0, 2, 1)
        cln = self.encoder[1](cln.reshape(B * F, cln.shape[2], 2).permute(0, 2, 1)).permute(0, 2, 1)
        # mix, cln = [self.encoder[i](v.reshape(B * F, v.shape[2], H0).permute(0, 2, 1)).permute(0, 2, 1) for i, v in enumerate([mix, cln])]
        H = mix.shape[2]
        mix, cln = mix.reshape(B, F, T, H), cln.reshape(B, F, cln.shape[1], H)

        attn_mask = get_local_mask(mix.shape[2], cln.shape[2], self.cross_mask_len / mix.shape[2], device=mix.device)

        for i in range(self.num_layers):
            if self.activation_checkpointing and self.training:
                mix, cln = cp.checkpoint(self.layer_forward, mix, cln, attn_mask, i, use_reentrant=False)
            else:
                mix, cln = self.layer_forward(mix, cln, attn_mask, i)

        if self.output == 'mix':
            y = self.decoder(mix)
        elif self.output == 'cln':
            y = self.decoder(cln)
        else:
            assert self.output == 'mix+cln', self.output
            y = self.decoder(torch.concat([mix, cln], dim=-1))
        return y

    def layer_forward(self, mix: Tensor, cln: Tensor, attn_mask: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        # FConv1
        mix = mix + self.sp_mix[i]._fconv(self.sp_mix[i].fconv1, mix)
        cln = cln + self.sp_cln[i]._fconv(self.sp_cln[i].fconv1, cln)
        # Full-Linear
        mix = mix + self.sp_mix[i]._full(mix)
        cln = cln + self.sp_cln[i]._full(cln)
        # FConv2
        mix = mix + self.sp_mix[i]._fconv(self.sp_mix[i].fconv2, mix)
        cln = cln + self.sp_cln[i]._fconv(self.sp_cln[i].fconv2, cln)
        # SA
        if '-mhsa' not in self.cfg_mix:
            mix = mix + self.sp_mix[i]._tsa(mix)[0]
        assert '-mhsa' in self.cfg_cln, self.cfg_cln
        # CA
        assert '+mhca' in self.cfg_mix, self.cfg_mix
        mix = mix + self.sp_mix[i]._tca(mix, cln, attn_mask)[0]
        if '+mhca' in self.cfg_cln:
            cln = cln + self.sp_cln[i]._tca(cln, mix, attn_mask.T)[0]
        # TConvFFN
        mix = mix + self.sp_mix[i]._tconvffn(mix)
        cln = cln + self.sp_cln[i]._tconvffn(cln)
        return mix, cln


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNetAlign2
    x = torch.randn((1, 129, 251, 2)).cuda()  # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    cln = torch.randn((1, 129, 462, 2)).cuda()  # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    spatialnet_small = AlignSpatialNet().cuda()

    torch._dynamo.config.optimize_ddp = False  # fix this issue: https://github.com/pytorch/pytorch/issues/111279#issuecomment-1870641439
    torch._dynamo.config.cache_size_limit = 64

    torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
    torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.

    import time
    ts = time.time()
    # spatialnet_small = torch.compile(spatialnet_small, dynamic=True)
    y = spatialnet_small(x, cln)
    te = time.time()
    print(spatialnet_small)
    print(y.shape)
    print(te - ts)

    # spatialnet_small = spatialnet_small.to('meta')
    # x, cln = x.to('meta'), cln.to('meta')
    # from torch.utils.flop_counter import FlopCounterMode
    # with FlopCounterMode(spatialnet_small, display=False) as fcm:
    #     y = spatialnet_small(x, cln)
    #     flops_forward_eval = fcm.get_total_flops()
    #     res = y.sum()
    #     res.backward()
    #     flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    # params_eval = sum(param.numel() for param in spatialnet_small.parameters())
    # print(f"flops_forward={flops_forward_eval/4e9:.2f}G/s, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")
