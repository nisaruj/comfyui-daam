# ComfyUI DAAM

![Download Count Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2Fcomfyui-daam&query=downloads&style=flat-square&label=Downloads&color=green)

**ComfyUI custom nodes for [Diffusion Attentive Attribution Maps (DAAM)](https://github.com/castorini/daam)**

This extension enables visualization of cross-attention heatmaps within Stable Diffusion models, showing exactly which parts of the image correspond to specific words in the prompt — good for analyzing and debugging model outputs and LoRA training.

👉 Example workflows are available in [workflows](https://github.com/nisaruj/comfyui-daam/tree/main/workflows).

![Sample Workflow](https://github.com/nisaruj/comfyui-daam/blob/main/img/workflow.png)

This project was adapted from the [SD Web UI implementation](https://github.com/kousw/stable-diffusion-webui-daam).  Special thanks to [@kousw](https://github.com/kousw) for the original work!

## ✨ Features
- ✅ Positive / Negative prompt heatmap visualization
- ✅ SDXL and SD 1.5 support
- ✅ Batched input support

## 🚀 Installation

Now Available on [Comfy Registry](https://registry.comfy.org/nodes/comfyui-daam) and [Custom Node Manager](https://github.com/Comfy-Org/ComfyUI-Manager)!

### Manual Install

Clone this repo into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/nisaruj/comfyui-daam.git
```

Then install the required packages
```bash
cd comfyui-daam
python -s -m pip install -r requirements.txt
```

Restart ComfyUI.


## 🧩 DAAM Nodes

### `CLIPTextEncodeWithTokens`

Identical to `CLIPTextEncode` but also outputs the tokenized prompt required for the analysis.

![Node: CLIPTextEncodeWithTokens](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_clip.png)

### `KSamplerDAAM`

A hooked version of `KSampler`. During sampling, it records attention maps for later analysis.

**Outputs:**
- `latent` — standard latent output
- `pos_heatmaps` — positive prompt's raw heatmaps for input into the analyzer
- `neg_heatmaps` — negative prompt's raw heatmaps for input into the analyzer

![Node: KSamplerDAAM](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_sampler.png)

### `DAAMAnalyzer`

This node generates overlay heatmaps that show which parts of the image correspond to selected words in the prompt.

**Inputs:**
- `clip` — CLIP model used to encode the attention text
- `tokens` — from `CLIPTextEncodeWithTokens`
- `heatmaps` — from `KSamplerDAAM`
- `images` — the output images to overlay the heatmaps
- A **text box** for comma-separated words to generate heatmaps

**Output:**
- A batch of images with word-level heatmaps overlaid

![Node: DAAMAnalyzer](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_analyzer.png)


## 📷 Example Output

**Prompt:** A photo of corgi with a cowboy hat riding a skateboard in the park

**Attention words**: corgi,skateboard,hat,park

![DAAM Result](https://github.com/nisaruj/comfyui-daam/blob/main/img/preview.png)

## 📦 Changelog

- **0.3.0**
    - Negative prompt support
    - Code Refactor and Minor bug fixes

- **0.2.0**
    - Added support for batched inputs
    - SD 1.5 models compatibility
    - Bug fixes and stability improvements
- **0.1.1**
    - Initial Version with SDXL support
