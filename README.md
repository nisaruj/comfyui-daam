# ComfyUI DAAM

**ComfyUI custom nodes for [Diffusion Attentive Attribution Maps (DAAM)](https://github.com/castorini/daam)**

This extension enables visualization of cross-attention heatmaps within Stable Diffusion models, showing exactly which parts of the image correspond to specific words in the prompt.

This project was adapted from the [SD Web UI implementation](https://github.com/kousw/stable-diffusion-webui-daam).  
Special thanks to [@kousw](https://github.com/kousw) for the original work!

Tested on SDXL models. ComfyUI v0.3.27 and Rev. 2469 (Released on 2024-08-05)

## Sample Workflow

![Sample Workflow](https://github.com/nisaruj/comfyui-daam/blob/main/img/workflow.png)

## Installation

### Manual Install

Clone this repo into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/nisaruj/comfyui-daam.git

```

Then install the required packages
```
cd comfyui-daam
python -s -m pip install -r requirements.txt
```

Restart ComfyUI.


## DAAM Nodes

### `CLIPTextEncodeWithTokens`

Identical to `CLIPTextEncode` but also outputs the tokenized prompt required for the analysis.

![Node: CLIPTextEncodeWithTokens](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_clip.png)

### `KSamplerDAAM`

A hooked version of `KSampler`. During sampling, it records attention maps for later analysis.

**Outputs:**
- `latent` — standard latent output
- `heatmaps` — raw heatmaps for input into the analyzer

![Node: KSamplerDAAM](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_sampler.png)

### `DAAMAnalyzer`

This node generates overlay heatmaps that show which parts of the image correspond to selected words in your prompt.

**Inputs:**
- `clip` – CLIP model used to encode the attention text
- `tokens` — from `CLIPTextEncodeWithTokens`
- `heatmaps` — from `KSamplerDAAM`
- `images` — the output image to overlay the heatmaps
- A **text box** for comma-separated words to generate heatmaps

**Output:**
- A batch of images with word-level heatmaps overlaid

![Node: DAAMAnalyzer](https://github.com/nisaruj/comfyui-daam/blob/main/img/node_analyzer.png)


## Example Result

![DAAM Result](https://github.com/nisaruj/comfyui-daam/blob/main/img/preview.png)
