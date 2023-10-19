"""
Most of this code comes from the timm  library.
We tried to disentangle from the timm library version.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import logging
from functools import partial

import torch
import torch.nn as nn

import collections
from itertools import repeat


from .helpers.vit_helpers import (
    DropPath,
    trunc_normal_,
    load_pretrained,
)

_logger = logging.getLogger()


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


# cfg for passt_s_p16_s16_128_ap468
DEFAULT_CFG = {
    "url": "https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s16-ap.468.pt",
    "architecture": "passt_s_p16_s16_128_ap468",
    "pool_size": None,
    "interpolation": "bicubic",
    "fixed_input_size": True,
    "first_conv": "patch_embed.proj",
    "mean": IMAGENET_DEFAULT_MEAN,
    "std": IMAGENET_DEFAULT_STD,
    "input_size": (1, 128, 998),
    "crop_pct": 1.0,
    "classifier": None,  # ("head.1", "head_dist"),
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
}


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # if not (H == self.img_size[0] and W == self.img_size[1]):
        #    warnings.warn(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        # to do maybe replace weights
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PaSST(nn.Module):
    """

    Based on the implementation of Vision Transformer in timm library.
     Take a look at the get_model function, adapting the weights of pretrained imagenet models.

    """

    def __init__(
        self,
        s_patchout_t=0,
        s_patchout_f=0,
        img_size=(128, 998),
        patch_size=16,
        stride=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        """
        Args:
            s_patchout_t: structured Patchout time integer, number of columns to be removed from the patches grid
            s_patchout_f: structured Patchout Frequency integer, number of rows to be removed from the patches grid
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """

        print("s_patchout_t: \t", s_patchout_t)
        print("s_patchout_f: \t", s_patchout_f)
        print("img_size: \t", img_size)
        print("patch_size: \t", patch_size)
        print("stride: \t", stride)
        print("in_chans: \t", in_chans)
        print("embed_dim: \t", embed_dim)
        print("depth: \t", depth)
        print("num_heads: \t", num_heads)
        print("mlp_ratio: \t", mlp_ratio)
        print("qkv_bias: \t", qkv_bias)
        print("drop_rate: \t", drop_rate)
        print("attn_drop_rate: \t", attn_drop_rate)
        print("drop_path_rate: \t", drop_path_rate)
        print("embed_layer: \t", embed_layer)
        print("norm_layer: \t", norm_layer)
        print("act_layer: \t", act_layer)

        super().__init__()
        self.s_patchout_t = s_patchout_t
        self.s_patchout_f = s_patchout_f
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # PaSST
        # refer to https://arxiv.org/abs/2110.05069 Section 2
        self.new_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )  # for C and D tokens
        self.freq_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.patch_embed.grid_size[0], 1)
        )
        self.time_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, 1, self.patch_embed.grid_size[1])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.new_pos_embed, std=0.02)
        trunc_normal_(self.freq_new_pos_embed, std=0.02)
        trunc_normal_(self.time_new_pos_embed, std=0.02)
        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, get_intermediates=None):
        x = self.patch_embed(x)  # [b, e, f, t]
        B_dim, E_dim, F_dim, T_dim = x.shape  # slow

        # Adding Time/Freq information
        x = x + self.time_new_pos_embed
        x = x + self.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.training and self.s_patchout_t:
            # ([1, 768, 1, 82])
            random_indices = (
                torch.randperm(T_dim)[: T_dim - self.s_patchout_t].sort().values
            )
            x = x[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            # [1, 768, 12, 1]
            random_indices = (
                torch.randperm(F_dim)[: F_dim - self.s_patchout_f].sort().values
            )
            x = x[:, :, random_indices, :]

        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)

        # Add the C/D tokens
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        dist_token = (
            self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
        )
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = self.pos_drop(x)

        if get_intermediates:
            intermediate_results = [
                self.blocks[start:layer](x)
                for start, layer in zip([0] + get_intermediates, get_intermediates)
            ]
            x = self.blocks[get_intermediates[-1]:](intermediate_results[-1])
            x = self.norm(x)

            # reduce intermediates to their cls token
            intermediate_results = [self.norm(y)[:, 0] for y in intermediate_results]

            return x[:, 0], x[:, 1], intermediate_results

        else:
            x = self.blocks(x)
            x = self.norm(x)
            return x[:, 0], x[:, 1], []

    def forward(self, x):
        x = self.forward_features(x)
        features = (x[0] + x[1]) / 2
        return x, features


def _init_vit_weights(module: nn.Module, name: str = "", head_bias: float = 0.0):
    """ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def get_model(
    arch="passt_s_kd_p16_128_ap486",
    pretrained=True,
    in_channels=1,
    fstride=10,
    tstride=10,
    input_fdim=128,
    input_tdim=998,
    s_patchout_t=0,
    s_patchout_f=0,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
):
    """todo"""

    input_size = (input_fdim, input_tdim)
    stride = (fstride, tstride)

    if fstride != 16 or tstride != 16:
        raise ValueError(
            "fstride and tstride must be 16 for arch=passt_s_p16_s16_128_ap468. "
            "This model is pretrained with 16x16 patches and 16,16 strides."
            "Having different values will result in a different freq/time positional encoding shape."
            "you can solve this issue by calling get_model with "
            "get_model(arch='passt_s_p16_s16_128_ap468'...,fstride=16, tstride=16)"
        )

    print(
        "\n\nLoading PaSST pre-trained on AudioSet Patch 16 stride 16 structured patchout mAP=468 \n\n"
    )

    # Build the model
    model = PaSST(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        in_chans=in_channels,
        img_size=input_size,
        stride=stride,
        s_patchout_t=s_patchout_t,
        s_patchout_f=s_patchout_f,
    )
    model.default_cfg = DEFAULT_CFG

    if pretrained:
        load_pretrained(
            model,
            num_classes=-1,
            in_chans=1,
            strict=False,
        )

    return model
