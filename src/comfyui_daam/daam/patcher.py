from __future__ import annotations
from collections import defaultdict
from copy import deepcopy

from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import einsum

import math

from .util import UNetCrossAttentionLocator


class RawHeatMaps:
    def __init__(self):
        self.heat_maps = defaultdict(lambda: defaultdict(list))
        self.heat_maps_count = 0

        self.all_heat_maps = []

    def add_heat_map(self, batch_index, factor, heat_map):
        self.heat_maps[batch_index][factor].append(heat_map)
        self.heat_maps_count += 1

    def save_heat_map_timestep(self):
        if len(self.heat_maps) > 0:
            self.all_heat_maps.append(deepcopy(self.heat_maps))

        self.heat_maps.clear()
        self.heat_maps_count = 0

    def count(self):
        return self.heat_maps_count


class CrossAttentionPatcher:
    def __init__(
        self,
        model_patcher,
        img_height: int,
        img_width: int,
        context_size: tuple[int, int] = (77, 77),  # [0] for positive, [1] for negative
        enable_heat_maps: tuple[bool, bool] = (
            True,
            True,
        ),  # [0] for positive, [1] for negative
        weighted: bool = False,
        head_idx: int = 0,
    ):
        self.model_patcher = model_patcher
        self.img_height = img_height
        self.img_width = img_width
        self.heat_maps = {
            "pos": RawHeatMaps() if enable_heat_maps[0] else None,
            "neg": RawHeatMaps() if enable_heat_maps[1] else None,
        }
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.saved_model_options = None

    @property
    def all_heat_maps(self):
        pos_heat_maps = (
            self.heat_maps["pos"].all_heat_maps if self.heat_maps["pos"] else None
        )
        neg_heat_maps = (
            self.heat_maps["neg"].all_heat_maps if self.heat_maps["neg"] else None
        )
        return pos_heat_maps, neg_heat_maps

    def patch(self, layer_idx: int = None):
        self.block_tags = UNetCrossAttentionLocator().locate(
            self.model_patcher.model.diffusion_model, layer_idx=layer_idx
        )

        self.saved_model_options = self.model_patcher.model_options.copy()

        for block_name, layer_idx, transformer_index in self.block_tags:
            self.model_patcher.set_model_patch_replace(
                self._attn2_patched, "attn2", block_name, layer_idx, transformer_index
            )

    def unpatch(self):
        if hasattr(self, "block_tags"):
            for block_name, layer_idx, transformer_index in self.block_tags:
                self.model_patcher.set_model_patch_replace(
                    None, "attn2", block_name, layer_idx, transformer_index
                )

        if self.saved_model_options is not None:
            self.model_patcher.model_options = self.saved_model_options.copy()

    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method="bicubic"):
        # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
        Diffusion model.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.
            method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

        Returns:
            `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
        """
        weight = torch.full((factor, factor), 1 / factor**2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = int(math.sqrt((self.img_height * x.size(1)) / self.img_width))
        w = int(self.img_width * h / self.img_height)

        h_fix = w_fix = 64
        if h >= w:
            w_fix = int((w * h_fix) / h)
        else:
            h_fix = int((h * w_fix) / w)

        maps = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = 1

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

                if method == "bicubic":
                    map_ = F.interpolate(map_, size=(h_fix, w_fix), mode="bicubic")
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(
                        F.conv_transpose2d(map_, weight, stride=factor).squeeze(1)
                    )

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)

        if self.head_idx:
            maps = maps[:, self.head_idx : self.head_idx + 1, :, :]

        return (weights * maps).sum(1, keepdim=True).cpu()

    def _attn2_patched(self, q, context=None, value=None, extra_options={}, mask=None):
        k = context
        v = value

        n_heads = extra_options["n_heads"]
        dim_head = extra_options["dim_head"]
        attn_precision = extra_options["attn_precision"]
        cond_or_uncond = extra_options["cond_or_uncond"]

        b, sequence_length, _ = q.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, n_heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * n_heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = self._basic_attention_patched(
            q,
            k,
            v,
            b,
            sequence_length,
            n_heads,
            dim_head,
            mask,
            cond_or_uncond,
            attn_precision,
        )

        return out

    def _basic_attention_patched(
        self,
        q,
        k,
        v,
        batch_size,
        sequence_length,
        n_heads,
        dim_head,
        mask,
        cond_or_uncond,
        attn_precision,
    ):
        def calc_factor_base(w, h):
            z = max(w / 64, h / 64)
            factor_b = min(w, h) * z
            return factor_b

        factor_base = calc_factor_base(self.img_width, self.img_height)
        batch_size_attention = q.shape[0]
        slice_size = batch_size_attention // batch_size

        out = torch.zeros(
            (batch_size_attention, sequence_length, dim_head),
            device=q.device,
            dtype=q.dtype,
        )

        scale = dim_head**-0.5

        for batch_index in range(batch_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size

            # force cast to fp32 to avoid overflowing
            if attn_precision == torch.float32:
                sim_slice = (
                    einsum(
                        "b i d, b j d -> b i j",
                        q[start_idx:end_idx].float(),
                        k[start_idx:end_idx].float(),
                    )
                    * scale
                )
            else:
                sim_slice = (
                    einsum(
                        "b i d, b j d -> b i j",
                        q[start_idx:end_idx],
                        k[start_idx:end_idx],
                    )
                    * scale
                )

            if mask is not None:
                if mask.dtype == torch.bool:
                    # TODO: check if this bool part matches pytorch attention
                    mask = rearrange(mask, "b ... -> b (...)")
                    max_neg_value = -torch.finfo(sim_slice.dtype).max
                    mask = repeat(mask, "b j -> (b h) () j", h=n_heads)
                    sim_slice.masked_fill_(~mask, max_neg_value)
                else:
                    if len(mask.shape) == 2:
                        bs = 1
                    else:
                        bs = mask.shape[0]
                    mask = (
                        mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
                        .expand(batch_size, n_heads, -1, -1)
                        .reshape(-1, mask.shape[-2], mask.shape[-1])
                    )
                    sim_slice.add_(mask)

            factor = int(math.sqrt(factor_base // sim_slice.shape[1]))

            # Attention
            sim_slice = sim_slice.softmax(-1)

            out_slice = einsum(
                "b i j, b j d -> b i d", sim_slice.to(v.dtype), v[start_idx:end_idx]
            )

            heat_map_added = self._add_heat_map_if_matches(
                batch_index,
                factor,
                sim_slice,
                v,
                cond_or_uncond,
                batch_size,
            )

            if heat_map_added:
                for pos_or_neg_heat_maps in self.heat_maps.values():
                    if (
                        pos_or_neg_heat_maps is not None
                        and pos_or_neg_heat_maps.count() >= len(self.block_tags)
                    ):
                        # Flush the heatmaps for each timestep
                        pos_or_neg_heat_maps.save_heat_map_timestep()

            out[start_idx:end_idx] = out_slice

        del q, k

        out = (
            out.unsqueeze(0)
            .reshape(batch_size, n_heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, -1, n_heads * dim_head)
        )

        return out

    def _add_heat_map_if_matches(
        self,
        batch_index: int,
        factor: int,
        sim_slice: torch.Tensor,
        value: torch.Tensor,
        cond_or_uncond: list,
        batch_size: int,
    ):
        if factor < 1:
            return False

        factor //= 1

        # ComfyUI determines whether run cond/uncond in 2 batches or 1 batch for each, based on free memory.
        # This will affect how we collect the heatmaps, so we need to keep track the cond_or_uncond variable.
        # len(cond_or_uncond) = 1: either cond [0] or uncond [1]
        # len(cond_or_uncond) = 2: uncond and cond in the same run [1, 0]
        # https://github.com/comfyanonymous/ComfyUI/blob/92cdc692f47188e6e4c48c5666ac802281240a37/comfy/samplers.py#L260
        if len(cond_or_uncond) == 2 and sim_slice.shape[-1] == max(self.context_size):
            # Combined conditional/unconditional in the same batch
            half = batch_size // 2
            if batch_index >= half and self.heat_maps["pos"] is not None:
                # Conditional batch
                maps = self._up_sample_attn(sim_slice, value, factor)
                self.heat_maps["pos"].add_heat_map(batch_index - half, factor, maps)
            elif self.heat_maps["neg"] is not None:
                # Unconditional batch
                maps = self._up_sample_attn(sim_slice, value, factor)
                self.heat_maps["neg"].add_heat_map(batch_index, factor, maps)

            return True

        elif (
            sim_slice.shape[-1] == self.context_size[0]
            and cond_or_uncond == [0]
            and self.heat_maps["pos"] is not None
        ):
            # Separate conditional batch
            maps = self._up_sample_attn(sim_slice, value, factor)
            self.heat_maps["pos"].add_heat_map(batch_index, factor, maps)

            return True

        elif (
            sim_slice.shape[-1] == self.context_size[1]
            and cond_or_uncond == [1]
            and self.heat_maps["neg"] is not None
        ):
            # Separate unconditional batch
            maps = self._up_sample_attn(sim_slice, value, factor)
            self.heat_maps["neg"].add_heat_map(batch_index, factor, maps)

            return True

        return False
