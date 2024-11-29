import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


# Dual up-sample
class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        elif self.factor == 4:
            self.conv = nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution

        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution

        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))
        out = out.permute(0, 2, 3, 1)  # B, H, W, C
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        return out


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SUNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3

        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample", **kwargs):
        super(SUNet, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.prelu = nn.PReLU()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, residual, x_downsample

    def forward(self, x):
        x = self.conv_first(x)
        x, residual, x_downsample = self.forward_features(x)
        # x = x + residual
        return x, x_downsample

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops


class Decoder(nn.Module):
    def __init__(self, img_size=128, patch_size=4,
                 embed_dim=96, depths=[8, 8, 8, 8, 8],
                 num_heads=[8, 8, 8, 8, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, final_upsample="Dual up-sample"):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.layers_up = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), scale_factor=2)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "Dual up-sample":
            self.up = UpSample(input_resolution=(img_size // patch_size, img_size // patch_size),
                               in_channels=embed_dim, scale_factor=4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "Dual up-sample":
            x = self.up(x)
            # x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x, x_downsample):
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        # x = x + residual
        return x


class Fused(nn.Module):
    def __init__(self, img_size=128, patch_size=4,
                 embed_dim=96, depths=[8, 8, 8, 8],
                 num_heads=[8, 8, 8, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, final_upsample="Dual up-sample"):
        super(Fused, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.layers_up = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), scale_factor=2)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "Dual up-sample":
            self.up = UpSample(input_resolution=(img_size // patch_size, img_size // patch_size),
                               in_channels=embed_dim, scale_factor=4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # Dencoder and Skip connection
    def forward_up_features(self, x, z):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:

                x = layer_up(x)
            else:
                x = torch.cat([x, z[3 - inx]], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "Dual up-sample":
            x = self.up(x)
            # x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x, z):
        x = self.forward_up_features(x, z)
        x = self.up_x4(x)
        # x = x + residual
        return x


class Re_HS(nn.Module):
    def __init__(self, in_ch,  out_ch):
        super(Re_HS, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class Attention(nn.Module):
    def __init__(self, in_ch, input_resolution = (4, 4)):
        super(Attention, self).__init__()
        self.inn1 = INN(in_ch, input_resolution = input_resolution)
        #self.inn2 = INN(in_ch, input_resolution = input_resolution)
        # basic_block_layer = []
        # for _ in range(2):
        #     basic_block_layer += [SwinTransformerBlock(dim=in_ch * 3, input_resolution=input_resolution,
        #                          num_heads=4, window_size=8,
        #                          shift_size=0,
        #                          mlp_ratio=4.,
        #                          qkv_bias=True, qk_scale=2,
        #                          drop=0., attn_drop=0.,
        #                          drop_path=0.,
        #                          norm_layer=nn.LayerNorm),
        #                 nn.ReLU6(inplace = True)]
        # self.basic_block = nn.Sequential(*basic_block_layer)
        # self.P_C_f = nn.Sequential(*self.basic_block)

        self.trans = nn.Sequential(nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1),
                                   nn.ReLU6(inplace=True))

    def forward(self, x, y, z):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        x, y = self.inn1(x, y)
        # fused = fused.permute(0, 2, 3, 1)
        # fused = fused.reshape(B, -1, C)
        fused = torch.cat([x, y, z], dim = -1)
        # fused = self.P_C_f(fused)
        fused = fused.view(B, H, W, 3 * C)
        fused = fused.permute(0, 3, 1, 2)  # B, C, H, W
        fused = self.trans(fused)
        # fused = fused.permute(0, 2, 3, 1)
        # fused = fused.view(B, -1, C)
        # fused = self.inn2(fused, z)
        fused = fused.permute(0, 2, 3, 1)
        fused = fused.reshape(B, -1, C)
        return fused

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


def calculate_orthogonality_loss(first_feature, second_feature):
    diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(1, 2)), dim=(1, 2)).pow(2).mean()
    return diff_loss


def mutual_information_loss(x, y):
    B, H = x.shape
    loss = 0
    for i in range(B):
        Kx = torch.unsqueeze(x[i, ...], 0) - torch.unsqueeze(x[i, ...], 1)
        Kx = torch.exp(- Kx ** 2)  # 计算核矩阵
        Ky = torch.unsqueeze(y[i, ...], 0) - torch.unsqueeze(y[i, ...], 1)
        Ky = torch.exp(- Ky ** 2)  # 计算核矩阵
        Kxy = torch.matmul(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n ** 2 + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / n
        loss += h * n ** 2 / (n - 1) ** 2
    return torch.mean(loss)


class Domain_Separate(nn.Module):
    def __init__(self, dim, input_resolution = (4, 4)):
        super(Domain_Separate, self).__init__()

        self.common_feature_extractor = nn.Sequential(
                        SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                             num_heads=4, window_size=8,
                                             shift_size=0,
                                             mlp_ratio=4.,
                                             qkv_bias=True, qk_scale=2,
                                             drop=0., attn_drop=0.3,
                                             drop_path=0.,
                                             norm_layer=nn.LayerNorm)
        )

        self.private_feature_extractor = nn.ModuleList([nn.Sequential(
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=4, window_size=8,
                                 shift_size=0,
                                 mlp_ratio=4.,
                                 qkv_bias=True, qk_scale=2,
                                 drop=0., attn_drop=0.,
                                 drop_path=0.,
                                 norm_layer=nn.LayerNorm),
        ) for _ in range(2)])

        self.modal_discriminator = nn.Sequential(
            nn.Linear(dim * input_resolution[0] * input_resolution[0], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.dimension_reduction = nn.Sequential(
            nn.Linear(dim * input_resolution[0] * input_resolution[0], 50),
        )
        self.adv_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        b, l, c = x.size()
        PAN_modal = torch.zeros(b).view(-1).cuda()
        HS_modal = torch.ones(b).view(-1).cuda()
        private_pan = self.private_feature_extractor[0](x)
        private_hs = self.private_feature_extractor[1](y)
        common_pan = self.common_feature_extractor(x)
        common_hs = self.common_feature_extractor(y)
        common = (common_pan + common_hs) / 2
        private_pan_modal_pred = self.modal_discriminator(private_pan.view(b, -1)).view(-1, 2)
        private_hs_modal_pred = self.modal_discriminator(private_hs.view(b, -1)).view(-1, 2)
        common_pan_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_pan.view(b, -1), 1)).view(-1, 2)
        common_hs_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_hs.view(b, -1), 1)).view(-1, 2)
        dimension_reduction_pan = self.dimension_reduction(private_pan.view(b, -1))
        dimension_reduction_hs = self.dimension_reduction(private_hs.view(b, -1))
        dimension_reduction_common = self.dimension_reduction(common.view(b, -1))
        private_diff_loss = calculate_orthogonality_loss(dimension_reduction_pan.unsqueeze(1), dimension_reduction_hs.unsqueeze(1))
        common_diff_loss = calculate_orthogonality_loss(dimension_reduction_pan.unsqueeze(1), dimension_reduction_common.unsqueeze(1)) + calculate_orthogonality_loss(dimension_reduction_hs.unsqueeze(1), dimension_reduction_common.unsqueeze(1))
        private_diff_loss1 = mutual_information_loss(dimension_reduction_pan, dimension_reduction_hs)
        common_diff_loss1 = mutual_information_loss(dimension_reduction_pan,
                                                    dimension_reduction_common) + mutual_information_loss(
            dimension_reduction_hs, dimension_reduction_common)
        adv_private_loss = self.adv_loss(private_pan_modal_pred, PAN_modal.long()) + self.adv_loss(
            private_hs_modal_pred, HS_modal.long())
        adv_common_loss = self.adv_loss(common_pan_modal_pred, PAN_modal.long()) + self.adv_loss(common_hs_modal_pred,
                                                                                                 HS_modal.long())
        loss = 0.01 * (adv_common_loss + adv_private_loss) + 5e-6 * (
                    private_diff_loss + common_diff_loss + private_diff_loss1 + common_diff_loss1)
        return loss, private_pan.view(b, l, c), private_hs.view(b, l, c), common.view(b, l, c)

# class P_C_fuse(nn.Module):
#     def __init__(self, in_ch, input_resolution = (4, 4)):
#         super(P_C_fuse, self).__init__()
#         self.P_C_f = nn.Sequential(
#                     SwinTransformerBlock(dim=in_ch * 2, input_resolution=input_resolution,
#                                  num_heads=4, window_size=8,
#                                  shift_size=0,
#                                  mlp_ratio=4.,
#                                  qkv_bias=True, qk_scale=2,
#                                  drop=0., attn_drop=0.,
#                                  drop_path=0.,
#                                  norm_layer=nn.LayerNorm),
#                     nn.ReLU6(inplace = True))
#
#         self.trans = nn.Sequential(nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1),
#                                    nn.ReLU6(inplace=True))
#         # self.inn = INN(in_ch, input_resolution = input_resolution)
#
#     def forward(self, x, y):
#         B, L, C = x.shape
#         H, W = int(math.sqrt(L)), int(math.sqrt(L))
#         fused = torch.cat([x, y], dim = -1)
#         fused = self.P_C_f(fused)
#         fused = fused.view(B, H, W, C * 2)
#         fused = fused.permute(0, 3, 1, 2)  # B, C, H, W
#         fused = self.trans(fused)
#         fused = fused.permute(0, 2, 3, 1)
#         fused = fused.reshape(B, -1, C)
#         return fused

class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()
        self.PAN = SUNet(img_size=128, patch_size=4, in_chans=1, out_chans=3,
                         embed_dim=64, depths=[8, 8, 8, 8],
                         num_heads=[4, 4, 4, 4],
                         window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         use_checkpoint=False, final_upsample="Dual up-sample")  # .cuda()
        # print(model)
        self.HS = SUNet(img_size=128, patch_size=4, in_chans=102, out_chans=3,
                        embed_dim=64, depths=[8, 8, 8, 8],
                        num_heads=[4, 4, 4, 4],
                        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, final_upsample="Dual up-sample")  # .cuda()

        self.de_PAN = Decoder(img_size=128, patch_size=4,
                               embed_dim=64, depths=[8, 8, 8, 8],
                               num_heads=[4, 4, 4, 4],
                               window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, use_checkpoint=False, final_upsample="Dual up-sample")

        self.de_HS = Decoder(img_size=128, patch_size=4,
                               embed_dim=64, depths=[8, 8, 8, 8],
                               num_heads=[4, 4, 4, 4],
                               window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, use_checkpoint=False, final_upsample="Dual up-sample")

        self.fused = Fused(img_size=128, patch_size=4,
                           embed_dim=64, depths=[8, 8, 8, 8],
                           num_heads=[4, 4, 4, 4],
                           window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                           norm_layer=nn.LayerNorm, use_checkpoint=False, final_upsample="Dual up-sample")

        self.ds1 = Domain_Separate(512, (4, 4))
        self.ds2 = Domain_Separate(64, (32, 32))
        self.ds3 = Domain_Separate(128, (16, 16))
        self.ds4 = Domain_Separate(256, (8, 8))
        # self.ds3 = Domain_Separate()

        self.att1 = Attention(512, (4, 4))
        self.att2 = Attention(64, (32, 32))
        self.att3 = Attention(128, (16, 16))
        self.att4 = Attention(256, (8, 8))

        self.re_pan = Re_HS(in_ch=64, out_ch=1)
        self.re_hs = Re_HS(in_ch=64, out_ch=102)
        self.re = Re_HS(in_ch=64, out_ch=102)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y):
        z = []
        out1, downsample1 = self.PAN(x)
        out2, downsample2 = self.HS(y)

        loss1, p1_1, p1_2, c1 = self.ds1(out1, out2)
        loss2, p2_1, p2_2, c2 = self.ds2(downsample1[0], downsample2[0])
        loss3, p3_1, p3_2, c3 = self.ds3(downsample1[1], downsample2[1])
        loss4, p4_1, p4_2, c4 = self.ds4(downsample1[2], downsample2[2])

        out = self.att1(p1_1, p1_2, c1)
        z.append(self.att2(p2_1, p2_2, c2))
        z.append(self.att3(p3_1, p3_2, c3))
        z.append(self.att4(p4_1, p4_2, c4))
        out1 = self.de_PAN(out1, downsample1)
        out2 = self.de_HS(out2, downsample2)
        pan = self.re_pan(out1)
        hs = self.re_hs(out2)
        out = self.fused(out, z)
        out = self.re(out)
        return out, pan, hs, (loss1 + loss2 + loss3 + loss4) / 4

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, input_resolution = (4, 4)):
        super(InvertedResidualBlock, self).__init__()
        self.bottleneckBlock = nn.Sequential(SwinTransformerBlock(dim=inp, input_resolution=input_resolution,
                             num_heads=4, window_size=8,
                             shift_size=0,
                             mlp_ratio=4.,
                             qkv_bias=True, qk_scale=2,
                             drop=0., attn_drop=0.,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm),
                            nn.ReLU6(inplace=True))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.bottleneckBlock(x)
        return x


class DetailNode(nn.Module):
    def __init__(self, channels, input_resolution = (4, 4)):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=channels, input_resolution = input_resolution)
        self.theta_rho = InvertedResidualBlock(inp=channels, input_resolution = input_resolution)
        self.theta_eta = InvertedResidualBlock(inp=channels, input_resolution = input_resolution)
        self.shffleconv = InvertedResidualBlock(inp=channels * 2, input_resolution = input_resolution)

    def separateFeature(self, x):
        z1, z2 = x[:, :, :x.shape[2] // 2], x[:, :, x.shape[2] // 2:x.shape[2]]
        return z1, z2


    def forward(self, z1, z2):
        z1 = z1.to(torch.float32)
        z2 = z2.to(torch.float32)
        y = torch.cat([z1, z2], dim=-1)
        y = self.shffleconv(y)
        z1, z2 = self.separateFeature(y)
        z3 = self.theta_phi(z1)
        z2 = z2 + z3
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class INN(nn.Module):
    def __init__(self, channels, input_resolution = (4, 4), num_layers=2):
        super(INN, self).__init__()
        INNmodules = [DetailNode(channels, input_resolution) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        #self.conv = nn.Conv2d(channels*2, channels, 1, 1)

    def forward(self, x, y):
        for layer in self.net:
            z1, z2 = layer(x, y)
        # B, L, C = x.shape
        # H, W = int(math.sqrt(L)), int(math.sqrt(L))
        # z1 = z1.view(B, H, W, C)
        # z1 = z1.permute(0, 3, 1, 2)  # B, C, H, W
        # z2 = z2.view(B, H, W, C)
        # z2 = z2.permute(0, 3, 1, 2)  # B, C, H, W
        return z1, z2



if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device = torch.device("cuda")
    height = 128
    width = 128
    x = torch.randn((1, 1, height, width))  # .cuda()
    y = torch.randn((1, 102, height, width))  # .cuda()
    model = nn.DataParallel(Reconstruction()).to(device)
    print(model(x, y)[0].shape)