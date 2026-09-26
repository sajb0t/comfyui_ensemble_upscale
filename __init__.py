# -*- coding: utf-8 -*-
"""
ComfyUI custom node package: Smart Ensemble Upscale 🔬

Exposes the node mappings that ComfyUI looks for when loading a custom node
package from ``ComfyUI/custom_nodes/``.
"""

from .smart_upscale_node import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Tell ComfyUI these two names are the public API of the package.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
