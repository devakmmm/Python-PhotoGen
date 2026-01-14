# Realistic Shadow Generator

Python CLI that composites a foreground subject onto a background with a realistic directional shadow. It generates a silhouette-based shadow (not a drop-shadow), adds a tight contact shadow, and increases blur/opacity falloff with distance. Optional depth maps can warp the shadow for uneven surfaces.

## Requirements

- Python 3.10+
- `Pillow`, `numpy`

Install deps:

```bash
python3 -m pip install pillow numpy
```

## Quick Start

```bash
python3 shadow_generator.py \
  --foreground "25_1107O_11974 PB + 1 - Photo Calendar B_Lamborghini HAS.JPG" \
  --background "B_Lamborghini Red.JPG" \
  --mask-threshold 25 \
  --angle 320 \
  --elevation 35 \
  --falloff 260 \
  --shadow-opacity 0.85 \
  --contact-strength 1.1 \
  --contact-threshold 18 \
  --output composite.png \
  --shadow-only shadow_only.png \
  --mask-debug mask_debug.png
```

## Outputs

- `composite.png`: final composite
- `shadow_only.png`: shadow intensity map (grayscale)
- `mask_debug.png`: foreground cutout mask

## Parameters (key ones)

- `--angle`: light direction in degrees (0–360)
- `--elevation`: light elevation in degrees (0–90); lower = longer shadows
- `--falloff`: distance for opacity decay (pixels)
- `--shadow-opacity`: global shadow opacity
- `--contact-strength`: strength of contact shadow at the base
- `--contact-threshold`: max distance (pixels) for contact shadow band
- `--depth`: optional grayscale depth map (0–255) to warp the shadow
- `--depth-strength`: warp strength (pixels)
- `--mask-threshold`: override auto mask threshold if needed
- `--mask-blur`: soften mask edges

## How it works (summary)

- Foreground cutout: estimates background color from the border and thresholds color distance; falls back to k-means color clustering when coverage is off. The mask is cleaned with morphology and hole filling.
- Shadow projection: builds a height map from the subject mask, then uses an affine shear based on light angle/elevation to cast the silhouette across the ground plane.
- Realism: shadow opacity decays with distance; blur radius increases with distance; a contact band strengthens the base.
- Depth map (optional): shifts the shadow along the light direction by local depth values to bend over uneven surfaces.

## Notes

- If your foreground already has transparency (cutout PNG), it uses the alpha channel directly.
- For tricky cutouts, tweak `--mask-threshold` or provide a pre-cutout PNG.
