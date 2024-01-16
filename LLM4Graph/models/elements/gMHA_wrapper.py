import math
import torch.nn as nn 
import torch
from typing import Optional, Tuple
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        batch_first=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        A: Optional[Tensor] = None,
        numerical_stability: bool = False,
        gt_style: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0)
                                 for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v
        
        # numerical stability
        if numerical_stability:
            # numerical stability
            max_val = attn_weights.max(dim=-1, keepdim=True)[0]
            attn_weights = torch.exp(attn_weights - max_val)
            # attn_weights = torch.exp(attn_weights)

            if not gt_style:
                attn_weights = attn_weights / \
                attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            


        if A is not None:
            A = torch.repeat_interleave(A, repeats=self.num_heads, dim=0)
            attn_weights = attn_weights * A
        
        if gt_style:
            attn_weights = attn_weights / \
                attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if self.batch_first and is_batched:
            attn = attn.transpose(1, 0)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GraphormerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GraphormerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, self_attention=True, batch_first=batch_first)
        self.spatial_pos_encoder = nn.Linear(1, nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None, A=None):
        x = src
        attn_bias = None
        if A is not None:
            # spatial pos
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            attn_bias = self.spatial_pos_encoder(
                A.unsqueeze(-1)).permute(0, 3, 1, 2)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask, attn_bias=attn_bias)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class HadamardEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HadamardEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, self_attention=True, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    # self-attention block

    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias, A):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           A=A, numerical_stability = True)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, attn_bias=None, A=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A))
            x = self.norm2(x + self._ff_block(x))
        return x

class GTEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GTEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, self_attention=True, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.num_head = nhead

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, attn_bias, A):
        x = self.self_attn(x, x, x,
                           attn_bias=attn_bias,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           A=A, 
                           gt_style=True)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, attn_bias=None, A=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, attn_bias=attn_bias, A=A))
            x = self.norm2(x + self._ff_block(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p d -> b d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d p -> b p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x



class MLPMixer(nn.Module):
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


class Hadamard(nn.Module):
    # Hadamard attention (default): (A ⊙ softmax(QK^T/sqrt(d)))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([HadamardEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class Standard(nn.Module):
    # standard (full) attention: softmax(QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=None, src_key_padding_mask=mask)
        return x


class Graph(nn.Module):
    # Graph attention (GT-like): softmax(A ⊙ QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class Kernel(nn.Module):
    # Kernel attention (GraphiT-like): softmax(random_walk(A) ⊙ QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x


class Addictive(nn.Module):
    # Addictive attention (Graphormer-like): softmax(QK^T/sqrt(d))V + LL(A)
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GraphormerEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x