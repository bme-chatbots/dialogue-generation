"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-kg
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import torch


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers=2, layer_cls=None, norm=None, **kwargs):
        super().__init__()

        if layer_cls is None:
            layer_cls = TransformerEncoderLayer

        layers = [layer_cls(**kwargs) for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
        out = src

        for layer in self.layers:
            out = layer(
                src=out,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                **kwargs
            )

        if self.norm is not None:
            out = self.norm(out)

        return out


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=4,
        d_ff=None,
        dropout=0.1,
        linear_cls=None,
        attn_cls=None,
        norm_cls=None,
        **kwargs
    ):
        super().__init__()

        if attn_cls is None:
            attn_cls = RelativeMultiheadAttention

        attn_kwargs = get_kwargs("attn_", kwargs)

        self.self_attn = attn_cls(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, **attn_kwargs
        )

        if linear_cls is None:
            linear_cls = torch.nn.Linear

        linear_kwargs = get_kwargs("linear_", kwargs)

        if d_ff is None:
            d_ff = d_model * 4

        self.linear1 = linear_cls(
            in_features=d_model, out_features=d_ff, **linear_kwargs
        )
        self.linear2 = linear_cls(
            in_features=d_ff, out_features=d_model, **linear_kwargs
        )

        if norm_cls is None:
            norm_cls = torch.nn.LayerNorm

        norm_kwargs = get_kwargs("norm_", kwargs)

        self.norm1 = norm_cls(d_model, **norm_kwargs)
        self.norm2 = norm_cls(d_model, **norm_kwargs)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2, _ = self.self_attn(
            key=src,
            query=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            **kwargs
        )

        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))

        return src


class RelativeMultiheadAttention(torch.nn.MultiheadAttention):
    def forward(
        self,
        query,
        key,
        value,
        position_embed,
        type_embed,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        tgt_len, bsz, embed_dim = query.size()
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // self.num_heads
        assert (
            head_dim * self.num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # self-attention
        p = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = p.chunk(3, dim=-1)

        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        q = q.contiguous().view(tgt_len, bsz, self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz, self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz, self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = relative_attention_inner_forward(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 3, 1),
            type_embed,
            position_embed.transpose(-2, -1),
        )

        assert list(attn_output_weights.size()) == [
            bsz,
            self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = relative_attention_inner_forward(
            attn_output_weights,
            v.permute(0, 2, 1, 3),
            type_embed.transpose(-2, -1),
            position_embed,
        ).view(bsz * self.num_heads, tgt_len, head_dim)

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = torch.nn.functional.linear(
            attn_output, self.out_proj.weight, self.out_proj.bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None
        tgt_len, bsz, embed_dim = query.size()
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // self.num_heads
        assert (
            head_dim * self.num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # self-attention
        p = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = p.chunk(3, dim=-1)

        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        q = q.contiguous().view(tgt_len, bsz, self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz, self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz, self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = relative_attention_inner_forward(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 3, 1),
            type_embed,
            position_embed,
        )

        assert list(attn_output_weights.size()) == [
            bsz,
            self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = relative_attention_inner_forward(
            attn_output_weights,
            v.permute(0, 2, 1, 3),
            type_embed.transpose(-2, -1),
            position_embed,
        ).view(bsz * self.num_heads, tgt_len, head_dim)

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = torch.nn.functional.linear(
            attn_output, self.out_proj.weight, self.out_proj.bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


def relative_attention_inner_forward(x, y, type_embed, position_embed):
    batch_size, heads, length, _ = x.size()

    x_t_r = x.permute(2, 0, 1, 3).reshape([length, heads * batch_size, -1])
    p = torch.matmul(x_t_r, position_embed)
    p_r_t = p.reshape([length, batch_size, heads, -1]).permute(1, 2, 0, 3)

    r = torch.matmul(type_embed, x.unsqueeze(-1)).squeeze(-1)

    return torch.matmul(x, y) + r + p_r_t


def create_relative_distance_matrix(length, distance):
    range_mat = torch.arange(length).expand(length, length)
    distance_mat = range_mat - range_mat.T

    distance_mat = torch.clamp(distance_mat, min=-distance, max=distance)

    return distance_mat + distance


def get_kwargs(prefix, kwargs):
    return {
        key.replace(prefix, ""): value
        for key, value in kwargs.items()
        if key.startswith(prefix)
    }
