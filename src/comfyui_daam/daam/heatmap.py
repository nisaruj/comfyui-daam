from __future__ import annotations
from comfy import model_management

from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm
import numpy as np
import torch
import torch.nn.functional as F
from os import path

__all__ = ["GlobalHeatMap", "HeatMapProcessor"]


class GlobalHeatMap:
    def __init__(self, prompt_analyzer, heat_maps: torch.Tensor, batch_index):
        self.prompt_analyzer = prompt_analyzer
        self.heat_maps = self.compute_global_heat_map(heat_maps, batch_index)

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> torch.Tensor:
        if word_idx:
            return self.heat_maps[word_idx]

        merge_idxs = self.prompt_analyzer.calc_word_indices(word)

        if len(merge_idxs) == 0:
            return None

        return self.heat_maps[merge_idxs].mean(0)

    def compute_global_heat_map(
        self,
        all_heat_maps,
        batch_index,
        time_weights=None,
        time_idx=None,
        last_n=None,
        first_n=None,
        factors=None,
    ):
        if len(all_heat_maps) == 0:
            return None

        if time_weights is None:
            time_weights = [1.0] * len(all_heat_maps)

        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum()

        if time_idx is not None:
            heat_maps = [all_heat_maps[time_idx]]
        else:
            heat_maps = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps
            heat_maps = heat_maps[:first_n] if first_n is not None else heat_maps

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []

        for batch_to_heat_maps in heat_maps:
            if batch_index not in batch_to_heat_maps:
                continue

            merge_list = []

            factors_to_heat_maps = batch_to_heat_maps[batch_index]

            for k, heat_map in factors_to_heat_maps.items():
                # heat_map shape: (tokens, 1, height, width)
                # each v is a heat map tensor for a layer of factor size k across the tokens
                if k in factors:
                    merge_list.append(torch.stack(heat_map, 0).mean(0))

            if len(merge_list) > 0:
                all_merges.append(merge_list)

        maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        maps = maps.sum(0).to(model_management.intermediate_device()).sum(2).sum(0)

        return maps


class HeatMapProcessor:
    def expand_image(
        im: torch.Tensor, h=512, w=512, absolute: bool = False, threshold: float = None
    ) -> torch.Tensor:
        im = im.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(h, w), mode="bicubic")

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        return im.squeeze()

    def expand_images(
        im: torch.Tensor,
        h=512,
        w=512,
        norm: str = "z-score",
        threshold: float = None,
        max_token_len=None,
    ) -> torch.Tensor:
        if im is None:
            return im

        # If a single image is provided, delegate to the existing single-image method
        if im.dim() == 2:
            return HeatMapProcessor.expand_image(
                im, h=h, w=w, absolute=absolute, threshold=threshold
            )

        if im.dim() != 3:
            raise ValueError(
                f"Expected im of shape (tokens, H, W) or (H, W), got {tuple(im.shape)}"
            )

        # im: (tokens, H, W) -> (tokens, 1, H, W) for interpolation
        im = im.unsqueeze(1)
        im = F.interpolate(im.float().detach(), size=(h, w), mode="bicubic")

        # If max_token_len is specified, set the value of all tokens beyond that to 0
        # Useful for truncating end tokens
        if max_token_len is not None and im.shape[0] > max_token_len:
            im[max_token_len:] = 0

        if norm != None:
            if norm == "min-max":
                mins = im.amin(dim=(-2, -1), keepdim=True)
                maxs = im.amax(dim=(-2, -1), keepdim=True)
                im = (im - mins) / (maxs - mins + 1e-8)
            elif norm == "z-score":
                mean = im.mean(dim=(2, 3), keepdim=True)
                std = im.std(dim=(2, 3), keepdim=True)
                im = (im - mean) / (std + 1e-8)

        if threshold is not None:
            im = (im > threshold).float()

        # return as (tokens, h, w)
        return im.squeeze(1)

    def _write_on_image(img, caption, font_size=36):
        ix, iy = img.size
        draw = ImageDraw.Draw(img)
        margin = 2
        draw = ImageDraw.Draw(img)

        base_dir = path.dirname(__file__)
        font_path = path.abspath(
            path.join(base_dir, "..", "fonts", "Roboto-Medium.ttf")
        )

        font = ImageFont.truetype(font_path, font_size)

        text_height = iy - 60
        tx = draw.textbbox((0, 0), caption, font)
        draw.text(
            (int((ix - tx[2]) / 2), text_height + margin), caption, (0, 0, 0), font=font
        )
        draw.text(
            (int((ix - tx[2]) / 2), text_height - margin), caption, (0, 0, 0), font=font
        )
        draw.text(
            (int((ix - tx[2]) / 2 + margin), text_height), caption, (0, 0, 0), font=font
        )
        draw.text(
            (int((ix - tx[2]) / 2 - margin), text_height), caption, (0, 0, 0), font=font
        )
        draw.text(
            (int((ix - tx[2]) / 2), text_height), caption, (255, 255, 255), font=font
        )
        return img

    def image_overlay_heat_map(
        base_img,
        heat_map,
        word=None,
        out_file=None,
        crop=None,
        alpha=0.5,
        caption=None,
        image_scale=1.0,
    ):
        # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
        assert base_img is not None

        if isinstance(base_img, torch.Tensor):
            if base_img.dim() == 4:
                base_img = base_img[0]  # Take first in batch
            # [H, W, C] in 0–255
            base_img = (base_img * 255).clamp(0, 255).byte()
            base_img = base_img.cpu().numpy()  # Convert to NumPy
            base_img = Image.fromarray(base_img)

        if heat_map is not None:
            heat_map = HeatMapProcessor._convert_heat_map_colors(heat_map)
            heat_map = heat_map.to("cpu").detach().numpy().copy().astype(np.uint8)
            heat_map_img = Image.fromarray(heat_map)

            img = Image.blend(base_img, heat_map_img, alpha)
        else:
            img = base_img.copy()

        if caption:
            img = HeatMapProcessor._write_on_image(img, caption)

        if image_scale != 1.0:
            x, y = img.size
            size = (int(x * image_scale), int(y * image_scale))
            img = img.resize(size, Image.BICUBIC)

        img_array = np.array(img).astype(np.float32) / 255.0  # [H, W, C]
        img_tensor = torch.from_numpy(img_array)  # [H, W, C]

        return img_tensor

    def _convert_heat_map_colors(heat_map: torch.Tensor):
        def get_color(value):
            return np.array(cm.turbo(value / 255)[0:3])

        color_map = np.array([get_color(i) * 255 for i in range(256)])
        color_map = torch.tensor(color_map, device=heat_map.device, dtype=torch.float32)

        heat_map = (heat_map * 255).long()

        return color_map[heat_map]
