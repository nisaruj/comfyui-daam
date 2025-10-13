from __future__ import annotations
from collections import defaultdict
from typing import Type
from einops import rearrange

from comfy.ldm.modules.diffusionmodules.mmdit import JointBlock

from comfy.ldm.modules.attention import optimized_attention

import torch

from .base_patcher import BaseAttentionPatcher
from ..attention import attention_sub_quad_patched

from ..util import ObjectHooker, AggregateHooker


__all__ = ["SD3AttentionPatcher"]


class SD3AttentionPatcher(BaseAttentionPatcher):
    # TODO: Refactor the class since some functions share the same code with Flux
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
        super().__init__(
            model_patcher,
            img_height,
            img_width,
            context_size=context_size,
            enable_heat_maps=enable_heat_maps,
            weighted=weighted,
            head_idx=head_idx,
        )
        self.tracers = [
            sd3trace(
                model_patcher,
                img_height,
                img_width,
                heat_maps=self.heat_maps,
                context_size=context_size[0],
            )
        ]

    def patch(self):
        for tracer in self.tracers:
            tracer.hook()

    def unpatch(self):
        for tracer in self.tracers:
            tracer.unhook()


class ModelFunctionWrapper:
    def __init__(self):
        self.cond_or_uncond = None

    def model_wrapper(self, unet_apply_function, unet_params) -> torch.Tensor:
        c = unet_params["c"]

        self.cond_or_uncond = unet_params["cond_or_uncond"]

        output = unet_apply_function(unet_params["input"], unet_params["timestep"], **c)
        return output


class SD3DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
        self,
        model,
        heigth: int,
        width: int,
        heat_maps,
        context_size: tuple[int, int] = (77, 77),
        weighted: bool = False,
        layer_idx: int = None,
        head_idx: int = None,
    ):
        # batch index, factor, attention
        self.heat_maps = heat_maps
        self.model_wrapper = ModelFunctionWrapper()

        selected_blocks = model.model.diffusion_model.joint_blocks
        self.block_len = len(selected_blocks)

        modules = [
            JointBlockHooker(
                x,
                heigth,
                width,
                heat_maps,
                context_size=context_size,
                weighted=weighted,
                head_idx=head_idx,
                cond_or_uncond_tracker=self.model_wrapper,
                block_len=self.block_len,
            )
            for x in selected_blocks
        ]

        self.height = heigth
        self.width = width
        self.model = model

        super().__init__(modules)

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps

    def reset(self):
        map(lambda module: module.reset(), self.module)
        return self.forward_hook.all_heat_maps.clear()

    def _hook_impl(self):
        self.model.set_model_unet_function_wrapper(self.model_wrapper.model_wrapper)

        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        self.model.set_model_unet_function_wrapper(None)

        for h in self.module:
            h.unhook()


class JointBlockHooker(ObjectHooker[JointBlock]):
    def __init__(
        self,
        module: JointBlock,
        img_height: int,
        img_width: int,
        heat_maps: defaultdict(defaultdict),
        context_size: tuple[int, int] = (77, 77),
        weighted: bool = False,
        head_idx: int = 0,
        cond_or_uncond_tracker=None,
        block_len=19,
    ):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.img_height = img_height
        self.img_width = img_width
        self.cond_or_uncond_tracker = cond_or_uncond_tracker
        self.block_len = block_len

    def _forward(hk_self, self, *args, **kwargs):
        return hk_self.block_mixing(
            self,
            *args,
            context_block=self.context_block,
            x_block=self.x_block,
            **kwargs,
        )

    def block_mixing(hk_self, self, *args, use_checkpoint=True, **kwargs):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                hk_self._block_mixing, self, *args, use_reentrant=False, **kwargs
            )
        else:
            return hk_self._block_mixing(self, *args, **kwargs)

    def _block_mixing(
        hk_self, self, context, x, context_block, x_block, c, transformer_options={}
    ):
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)

        o = []
        for t in range(3):
            o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
        qkv = tuple(o)

        attn, attn_probs = attention_sub_quad_patched(
            qkv[0],
            qkv[1],
            qkv[2],
            heads=x_block.attn.num_heads,
        )

        # Process the attention probabilities
        height = hk_self.img_height // 16

        # Average two cross-attention probs
        # [24, 4250, 4250] -> [24, 4096, 154]
        context_len = context_qkv[0].shape[1]
        attn_probs = (
            attn_probs[:, context_len:, :context_len]
            + attn_probs[:, :context_len, context_len:].transpose(1, 2)
        ) * 0.5

        # [24, 4096, 154] -> [B, 154, 24, 64, 64]
        attn_probs = rearrange(
            attn_probs,
            "(batch heads) (height width) txt_tokens -> batch txt_tokens heads height width",
            heads=x_block.attn.num_heads,
            height=height,
        )
        # [B, 154, 24, 64, 64] -> [B, 154, 1, 64, 64]
        attn_probs = attn_probs.sum(dim=2, keepdim=True).cpu()

        hk_self._add_heat_map_if_matches(
            self,
            attn_probs,
            hk_self.cond_or_uncond_tracker.cond_or_uncond,
        )

        context_attn, x_attn = (
            attn[:, : context_qkv[0].shape[1]],
            attn[:, context_qkv[0].shape[1] :],
        )

        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)

        else:
            context = None
        if x_block.x_block_self_attn:
            attn2 = optimized_attention(
                x_qkv2[0],
                x_qkv2[1],
                x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
            x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
        else:
            x = x_block.post_attention(x_attn, *x_intermediates)
        return context, x

    def _hook_impl(self):
        self.monkey_patch("forward", self._forward)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))

    def _add_heat_map_if_matches(
        hk_self,
        self,
        attn_probs: torch.Tensor,  # [batch, txt_tokens, 1, height, width]
        cond_or_uncond: list,
    ):
        batch_size = attn_probs.shape[0]
        factor = 2

        for batch_index in range(batch_size):
            sim_slice = attn_probs[batch_index]

            heat_map_added = False

            if len(cond_or_uncond) == 2:
                # Combined conditional/unconditional in the same batch
                half = batch_size // 2
                if batch_index >= half and hk_self.heat_maps["pos"] is not None:
                    # Conditional batch
                    hk_self.heat_maps["pos"].add_heat_map(
                        batch_index - half, factor, sim_slice
                    )
                elif hk_self.heat_maps["neg"] is not None:
                    # Unconditional batch
                    hk_self.heat_maps["neg"].add_heat_map(
                        batch_index, factor, sim_slice
                    )

                heat_map_added = True
            elif cond_or_uncond == [0] and hk_self.heat_maps["pos"] is not None:
                # Separate conditional batch
                hk_self.heat_maps["pos"].add_heat_map(batch_index, factor, sim_slice)

                heat_map_added = True

            elif cond_or_uncond == [1] and hk_self.heat_maps["neg"] is not None:
                # Separate unconditional batch
                hk_self.heat_maps["neg"].add_heat_map(batch_index, factor, sim_slice)

                heat_map_added = True

            if heat_map_added:
                for pos_or_neg_heat_maps in hk_self.heat_maps.values():
                    if (
                        pos_or_neg_heat_maps is not None
                        and pos_or_neg_heat_maps.count() >= hk_self.block_len
                    ):
                        # Flush the heatmaps for each timestep
                        pos_or_neg_heat_maps.save_heat_map_timestep()


sd3trace: Type[SD3DiffusionHeatMapHooker] = SD3DiffusionHeatMapHooker
