import comfy.samplers
import comfy.model_base
from comfy.cli_args import args

import latent_preview
import torch
import json
import folder_paths
import os
import random
import numpy as np

from .daam import analyzer
from .daam.heatmap import GlobalHeatMap, HeatMapProcessor
from .daam.patcher import (
    CrossAttentionPatcher,
    FluxAttentionPatcher,
    SD3AttentionPatcher,
)
from .daam.util import is_output_connected

from PIL import Image
from PIL.PngImagePlugin import PngInfo


class CLIPTextEncodeWithTokens:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text to be encoded.",
                    },
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model used for encoding the text."},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "TOKENS")
    OUTPUT_TOOLTIPS = (
        "A conditioning containing the embedded text used to guide the diffusion model.",
        "List of tokens after tokenization.",
    )
    FUNCTION = "encode"

    CATEGORY = "daam-nodes"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], tokens)


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSamplerDAAM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to include in the image."
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to exclude from the image."
                    },
                ),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "HEATMAP", "HEATMAP")
    RETURN_NAMES = ("latent", "pos_heatmaps", "neg_heatmaps")
    OUTPUT_TOOLTIPS = ("The denoised latent.", "The heatmap data.")
    FUNCTION = "sample"

    CATEGORY = "daam-nodes"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        prompt=None,
        node_id=None,
    ):
        _, _, lh, lw = latent_image["samples"].shape

        img_height = lh * 8
        img_width = lw * 8

        _, pos_context_size, _ = positive[0][0].shape
        _, neg_context_size, _ = negative[0][0].shape

        # Optimization: Only collect heatmaps when needed
        enable_pos_heat_maps = is_output_connected(prompt, node_id, 1)
        enable_neg_heat_maps = is_output_connected(prompt, node_id, 2)

        if isinstance(model.model, comfy.model_base.Flux):
            patcher_class = FluxAttentionPatcher
        elif isinstance(model.model, comfy.model_base.SD3):
            patcher_class = SD3AttentionPatcher
        else:
            # SDXL & BaseModel (SD1.5)
            patcher_class = CrossAttentionPatcher

        patcher = patcher_class(
            model,
            img_height,
            img_width,
            context_size=(pos_context_size, neg_context_size),
            enable_heat_maps=(enable_pos_heat_maps, enable_neg_heat_maps),
        )

        patcher.patch()

        (latent_out,) = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )

        pos_heat_maps, neg_heat_maps = patcher.all_heat_maps

        patcher.unpatch()

        return (latent_out, pos_heat_maps, neg_heat_maps)


class DAAMAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model used for encoding the text."},
                ),
                "tokens": (
                    "TOKENS",
                    {
                        "tooltip": "The tokens used to encode the prompt.",
                        "forceInput": True,
                    },
                ),
                "heatmaps": (
                    "HEATMAP",
                    {"tooltip": "The heatmap data.", "forceInput": True},
                ),
                "attentions": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Attention words to analyze (Comma separated).",
                    },
                ),
                "caption": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to show the attention word as a caption.",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "The alpha value for the overlay.",
                    },
                ),
                "images": ("IMAGE", {"tooltip": "Output Images"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The heatmap image.",)
    FUNCTION = "analyze"

    CATEGORY = "daam-nodes"
    DESCRIPTION = "Generates a heatmap image from the attention maps and overlays it on the input images."

    def analyze(self, clip, tokens, heatmaps, attentions, caption, alpha, images=None):
        self.attentions = [
            word for word in map(str.strip, attentions.split(",")) if word
        ]

        self.prompt_analyzer = analyzer.PromptAnalyzer(clip, tokens)

        batch_count, img_height, img_width, _ = images.shape  # (B, H, W, C)

        embedded_imgs = []
        for batch_index in range(batch_count):
            # image: (H, W, C)
            image = images[batch_index]

            global_heat_map = GlobalHeatMap(self.prompt_analyzer, heatmaps, batch_index)

            for attention in self.attentions:
                heat_map = global_heat_map.compute_word_heat_map(attention)

                heat_map_img = (
                    HeatMapProcessor.expand_image(heat_map, img_height, img_width)
                    if heat_map is not None
                    else None
                )

                caption_text = attention if caption else None
                overlay_img: Image.Image = HeatMapProcessor.image_overlay_heat_map(
                    image,
                    heat_map_img,
                    attention,
                    alpha=alpha,
                    caption=caption_text,
                    image_scale=1,
                )

                embedded_imgs.append(overlay_img)

        embedded_imgs = torch.stack(embedded_imgs, dim=0)
        return (embedded_imgs,)


class DAAMPreviewImage:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "tokens": ("TOKENS",),
                "heatmaps": ("HEATMAP",),
                "images": ("IMAGE",),
                "top_tokens": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 10,
                        "tooltip": "The number of top tokens that will be displayed.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_and_display"

    OUTPUT_NODE = True

    CATEGORY = "daam-nodes"
    DESCRIPTION = "Preview the images along with attention heatmaps."

    def save_and_display(
        self,
        clip,
        tokens,
        heatmaps,
        images,
        top_tokens=5,
        filename_prefix="DAAMAttn",
        prompt=None,
        extra_pnginfo=None,
    ):
        self.prompt_analyzer = analyzer.PromptAnalyzer(clip, tokens)

        batch_count, img_height, img_width, _ = images.shape  # (B, H, W, C)
        
        top_attention_map = []
        word_len = self.prompt_analyzer.word_count()

        for batch_index in range(batch_count):
            # image: (H, W, C)
            # image = images[batch_index]

            global_heat_map = GlobalHeatMap(self.prompt_analyzer, heatmaps, batch_index)

            heat_maps = global_heat_map.heat_maps  # (num_tokens, 64, 64)

            expanded_heat_maps = (
                HeatMapProcessor.expand_images(
                    heat_maps, img_height, img_width, max_token_len=word_len
                )
                if heat_maps is not None
                else None
            )

            t_maps = expanded_heat_maps.detach().cpu()
            num_tokens = t_maps.shape[0]
            k = min(top_tokens, num_tokens)

            _, indices = torch.topk(t_maps, k, dim=0, largest=True, sorted=True)

            # Accepts maximum of 256 tokens for performance and storage reasons
            # Might consider increasing this limit in the future
            top_k_tokens = (
                indices.permute(1, 2, 0)
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )  # (Height, Width, K)

            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    self.output_dir,
                    images[0].shape[1],
                    images[0].shape[0],
                )
            )

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_index))
            file = f"{filename_with_batch_num}_{counter:05}.npy"
            np.save(os.path.join(full_output_folder, file), top_k_tokens)

            top_attention_map.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "temp",
                }
            )

        results = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
        decoded_texts = self.prompt_analyzer.get_decoded_texts()

        return {
            "ui": {
                "images": results,
                "heatmap": top_attention_map,
                "attention_words": decoded_texts,
            }
        }

    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return results


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KSamplerDAAM": KSamplerDAAM,
    "DAAMAnalyzer": DAAMAnalyzer,
    "CLIPTextEncodeWithTokens": CLIPTextEncodeWithTokens,
    "DAAMPreviewImage": DAAMPreviewImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerDAAM": "KSampler for DAAM",
    "DAAMAnalyzer": "DAAM Analyzer",
    "CLIPTextEncodeWithTokens": "CLIP Text Encode (With Tokens)",
    "DAAMPreviewImage": "DAAM Preview Image (Beta)",
}
