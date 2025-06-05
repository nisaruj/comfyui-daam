import comfy.samplers
import comfy.model_base

import latent_preview
import torch

from .daam import analyzer
from .daam.heatmap import GlobalHeatMap, HeatMapProcessor
from .daam.patcher import (
    CrossAttentionPatcher,
    FluxAttentionPatcher,
    SD3AttentionPatcher,
)
from .daam.util import is_output_connected

from PIL import Image


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

    CATEGORY = "conditioning"
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

    CATEGORY = "sampling"
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

    CATEGORY = "image"
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KSamplerDAAM": KSamplerDAAM,
    "DAAMAnalyzer": DAAMAnalyzer,
    "CLIPTextEncodeWithTokens": CLIPTextEncodeWithTokens,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerDAAM": "KSampler for DAAM",
    "DAAMAnalyzer": "DAAMAnalyzer",
    "CLIPTextEncodeWithTokens": "CLIP Text Encode (With Tokens)",
}
