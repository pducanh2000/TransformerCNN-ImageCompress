import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.swin_transformer.basic_blocks import DropPath, ResidualBlock, AttentionBlock


class WMSA(nn.Module):
    """
    Self attention module in SwinTransformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_head = self.input_dim // self.head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)

        # Change the shape of relative_position_params to (2*W-1, 2*W-1, n_heads)
        self.relative_position_params = self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, -1)

        # Change the shape of relative_position_params to (n_heads, 2*W-1, 2*W-1)
        # Using tensor.permute(2, 1, 0)???
        self.relative_position_params = nn.Parameter(self.relative_position_params.transpose(1, 2).transpose(0, 1))

    def generate_mask(self, h, w, p, shift):
        """
        Generating the mask of  Swin-MSA
        Args:
            h: number of windows on height
            w: number of windows on width
            p: window size
            shift: shift parameter for CyclicShift.
        Returns:
            attn_mask: should be (1, 1, w, p, p)
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True

        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        # h, w, p1, p2, p3, p4 = attn_mask.shape
        # attn_mask = attn_mask.view(h*w, p1*p2, p3*p4)  # attn_mask with the shape of (h*w, p*p, p*p)
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)   # attn_mask with the shape of (1, 1, h*w, p*p, p*p)

        return attn_mask

    def relative_embedding(self):
        # cord with the size (window_size * window_size, 2)
        cord = torch.tensor(np.array([i, j] for i in range(self.window_size) for j in range(self.window_size)))

        # relation with the shape of (Wd*Wd, None, 2) - (None, Wd*Wd, 2) = (Wd*Wd, Wd*Wd, 2)
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1

        # Negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

    def forward(self, x):
        """
        Forward pass of Window Multihead Self Attention module
        Args:
            x: input tensor with shape of (B, H, W, C)
        Returns:
            output: tensor shape (B, H, W, C)
        """
        if self.type != "W":
            # Roll on H, W dimensions with shift values equal to half of window size
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)   # Num of windows - h dimension
        w_windows = x.size(2)   # Num of windows - w dimension

        # square validation
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)   # qkv with the shape of (B, n_wins_h * n_wins_w, Wd*Wd, 3*attn_dim)

        # Get the q, k, v; shape of each tensor: (n_heads, B, n_wins_h * n_wins_w, Wd*Wd, attn_dim//n_heads)
        q, k, v = rearrange(qkv, 'b nw np (three_h c) -> three_h b nw np c', c=self.head_dim).chunk(3, dim=0)

        # Calculate the similarity in attention mechanism  (n_heads, B, num_windows, Wd*Wd)
        # Here the length of sequence is the window size * window_size
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale    # sim = q @ k.transpose(-2, -1) * self.scale

        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        # Using attn_mask to distinguish different subwindows
        if self.type != "W":
            # fill -inf where the value of attention mask is True
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)  # (n_heads, B, num_windows, Wd*Wd, Wd*Wd)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)   # output = probs @ v

        output = rearrange(output, 'h b w p c -> b w p (h c)')  # (B, num_windows, Wd*Wd, attn_dim)
        output = self.linear(output)
        if self.type != "W":
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dim=(1, 2))
        return output


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        """
        SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"], f'Type should be in ["W", "SW"] but got "{type} instead"'
        self.type = type

        print("Block Initial Type:{}, drop path rate {.6f}".format(self.type, drop_path))
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, output_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.layernorm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim)
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.layernorm1(x)))
        x = x + self.drop_path(self.msa(self.layernorm2(x)))
        return x


class ConvTransBlock(nn.Module):
    """
    SwinTransformer and Convolution block
    """
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type="W"):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path

        assert type in ["W", "SW"]
        self.type = type
        self.transformer_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, type)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        """
        Input shape x (B, in_ch, H, W)
        """
        out = self.conv1_1(x)   # (B, conv_dim + trans_dim, H, W)
        # Split the input on channel dimension
        conv_x, trans_x = torch.split(out, [self.conv_dim, self.trans_dim], dim=1)

        # Convolution branch
        conv_x = self.conv_block(conv_x)

        # Transformer branch
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.transformer_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path):
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type="W")
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type="SW")
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_row + 1, padding_col, padding_row + 1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)

        # Redundancy
        if resize:
            x = F.pad(x, (-padding_col, -padding_row-1, -padding_col, -padding_row-1))
        return trans_x


class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192):
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)

        if inter_dim is not None:
            self.in_conv = nn.Conv2d(input_dim, inter_dim, kernel_size=1, stride=1)
            self.out_conv = nn.Conv2d(inter_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)

        out += identity
        out = self.out_conv(out)
        return out
