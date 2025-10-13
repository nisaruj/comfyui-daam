"""Top-level package for comfyui_daam."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

from .src.comfyui_daam.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_daam.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./src/comfyui_daam/js"
