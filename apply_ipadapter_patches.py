"""Apply community PR #108 patches to ComfyUI-IPAdapter-Flux for ComfyUI v0.14+ compat.

The Shakker-Labs maintainer is unresponsive since 2025-06. PR #108 by redaah
fixes 2 breaking changes in current ComfyUI:
  1. flux/layers.py — flipped_img_txt attribute removed from DoubleStreamBlock
  2. utils.py — forward() now passes timestep_zero_index= and other kwargs

Reference: https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux/pull/108
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "ComfyUI-IPAdapter-Flux")

# Patch 1: flux/layers.py
layers = ROOT / "flux" / "layers.py"
src = layers.read_text()
old1 = "self.flipped_img_txt = original_block.flipped_img_txt"
new1 = "self.flipped_img_txt = getattr(original_block, 'flipped_img_txt', False)"
if old1 not in src:
    if new1 in src:
        print("layers.py: already patched, skip")
    else:
        sys.exit(f"layers.py: target not found, source may have changed")
else:
    layers.write_text(src.replace(old1, new1))
    print("layers.py: patched OK")

# Patch 2: utils.py
utils = ROOT / "utils.py"
src = utils.read_text()
old2 = (
    "    guidance: Tensor|None = None,\n"
    "    control=None,\n"
    "    transformer_options={},\n"
    "    attn_mask: Tensor = None,\n"
    ") -> Tensor:"
)
new2 = (
    "    guidance: Tensor|None = None,\n"
    "    control=None,\n"
    "    timestep_zero_index=None,\n"
    "    transformer_options={},\n"
    "    attn_mask: Tensor = None,\n"
    "    **kwargs,\n"
    ") -> Tensor:"
)
if old2 not in src:
    if "timestep_zero_index=None" in src:
        print("utils.py: already patched, skip")
    else:
        sys.exit(f"utils.py: target not found, source may have changed")
else:
    utils.write_text(src.replace(old2, new2))
    print("utils.py: patched OK")

print("All PR #108 patches applied successfully")
