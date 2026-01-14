#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from collections import deque


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def estimate_bg_color(img: Image.Image, border: int) -> np.ndarray:
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    b = max(1, min(border, min(h, w) // 10))
    top = arr[:b, :, :]
    bottom = arr[-b:, :, :]
    left = arr[:, :b, :]
    right = arr[:, -b:, :]
    samples = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    return np.median(samples, axis=0)


def otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.flatten(), minlength=256)
    total = values.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def kmeans_centroids(data: np.ndarray, k: int, iters: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if data.shape[0] < k:
        return data[:k]
    centroids = data[rng.choice(data.shape[0], k, replace=False)]
    for _ in range(iters):
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for i in range(k):
            members = data[labels == i]
            if members.size == 0:
                centroids[i] = data[rng.integers(0, data.shape[0])]
            else:
                centroids[i] = members.mean(axis=0)
    return centroids


def assign_labels_full(arr: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3).astype(np.float32)
    best_label = np.zeros(flat.shape[0], dtype=np.int32)
    best_dist = None
    for i, c in enumerate(centroids):
        diff = flat - c
        dist = np.sum(diff * diff, axis=1)
        if best_dist is None:
            best_dist = dist
            best_label[:] = i
        else:
            better = dist < best_dist
            best_label[better] = i
            best_dist[better] = dist[better]
    return best_label.reshape(h, w)


def kmeans_mask(img: Image.Image, k: int = 4) -> np.ndarray:
    max_side = 220
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        small = img.resize((int(img.width * scale), int(img.height * scale)), resample=Image.BILINEAR)
    else:
        small = img
    small_arr = np.asarray(small).astype(np.float32)
    data = small_arr.reshape(-1, 3)

    centroids = kmeans_centroids(data, k=k, iters=10, seed=1)

    dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels_small = np.argmin(dists, axis=1).reshape(small_arr.shape[:2])

    h, w = small_arr.shape[:2]
    b = max(1, min(12, min(h, w) // 10))
    border = np.zeros((h, w), dtype=bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    border_labels = labels_small[border]
    counts = np.bincount(border_labels.flatten(), minlength=len(centroids))
    total = counts.sum()
    order = np.argsort(counts)[::-1]
    bg_clusters = []
    acc = 0
    for idx in order:
        if counts[idx] == 0:
            continue
        bg_clusters.append(int(idx))
        acc += counts[idx]
        if acc / max(1, total) >= 0.9:
            break

    full_arr = np.asarray(img).astype(np.float32)
    labels_full = assign_labels_full(full_arr, centroids)
    mask = ~np.isin(labels_full, bg_clusters)
    return (mask.astype(np.uint8) * 255)


def fill_holes(mask: np.ndarray, max_side: int = 420) -> np.ndarray:
    h, w = mask.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = Image.fromarray(mask).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.NEAREST
        )
        mask_small = np.asarray(small)
    else:
        mask_small = mask

    fg = mask_small > 0
    bg = ~fg
    visited = np.zeros_like(bg, dtype=bool)
    q = deque()

    hs, ws = bg.shape
    for x in range(ws):
        if bg[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if bg[hs - 1, x]:
            visited[hs - 1, x] = True
            q.append((hs - 1, x))
    for y in range(hs):
        if bg[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if bg[y, ws - 1]:
            visited[y, ws - 1] = True
            q.append((y, ws - 1))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < hs and 0 <= nx < ws and bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    holes = bg & ~visited
    filled = fg | holes
    filled = filled.astype(np.uint8) * 255

    if scale < 1.0:
        filled = Image.fromarray(filled).resize((w, h), resample=Image.NEAREST)
        return np.asarray(filled)
    return filled


def largest_component(mask: np.ndarray, max_side: int = 360) -> np.ndarray:
    h, w = mask.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = Image.fromarray(mask).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.NEAREST
        )
        mask_small = np.asarray(small)
    else:
        mask_small = mask

    fg = mask_small > 0
    visited = np.zeros_like(fg, dtype=bool)
    best_component = np.zeros_like(fg, dtype=bool)
    best_count = 0
    hs, ws = fg.shape
    for y in range(hs):
        for x in range(ws):
            if not fg[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            component = []
            while q:
                cy, cx = q.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < hs and 0 <= nx < ws and fg[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(component) > best_count:
                best_count = len(component)
                best_component = np.zeros_like(fg, dtype=bool)
                for cy, cx in component:
                    best_component[cy, cx] = True

    if best_count == 0:
        return mask

    result = best_component.astype(np.uint8) * 255
    if scale < 1.0:
        result = Image.fromarray(result).resize((w, h), resample=Image.NEAREST)
        return np.asarray(result)
    return result


def choose_threshold(dist_norm: np.ndarray, desired: float = 0.35) -> int:
    candidates = []
    for perc in (55, 60, 65, 70, 75):
        t = int(np.percentile(dist_norm, perc))
        cov = float((dist_norm >= t).mean())
        candidates.append((t, cov))

    otsu = otsu_threshold(dist_norm)
    candidates.append((otsu, float((dist_norm >= otsu).mean())))

    valid = [(t, cov) for t, cov in candidates if 0.2 <= cov <= 0.65]
    if valid:
        best = min(valid, key=lambda item: abs(item[1] - desired))
        return int(best[0])

    best = min(candidates, key=lambda item: abs(item[1] - desired))
    return int(best[0])


def generate_mask(img: Image.Image, threshold: int | None, border: int, blur: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    b = max(1, min(border, min(img.size) // 10))
    bg = estimate_bg_color(img, b)
    diff = arr - bg[None, None, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    dist_norm = np.clip(dist / (dist.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    if threshold is None:
        threshold = choose_threshold(dist_norm)

    mask = (dist_norm >= threshold).astype(np.uint8) * 255
    coverage = mask.mean() / 255.0
    if coverage > 0.7:
        mask = 255 - mask
        coverage = mask.mean() / 255.0
    if coverage < 0.12 or coverage > 0.92:
        mask = kmeans_mask(img)
    mask = largest_component(mask)
    mask_img = Image.fromarray(mask)

    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))

    filled = fill_holes(np.asarray(mask_img))
    mask_img = Image.fromarray(filled)

    if blur > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask_img


def blur_by_distance(
    shadow_map: np.ndarray,
    dist_map: np.ndarray,
    bins: list[tuple[float, float]],
    radii: list[float],
) -> np.ndarray:
    soft_shadow = np.zeros_like(shadow_map, dtype=np.float32)
    for (d0, d1), radius in zip(bins, radii):
        band = (dist_map >= d0) & (dist_map < d1)
        if not np.any(band):
            continue
        layer = np.zeros_like(shadow_map, dtype=np.float32)
        layer[band] = shadow_map[band]
        layer_img = Image.fromarray(np.clip(layer * 255.0, 0, 255).astype(np.uint8))
        if radius > 0:
            layer_img = layer_img.filter(ImageFilter.GaussianBlur(radius=radius))
        layer_arr = np.asarray(layer_img).astype(np.float32) / 255.0
        soft_shadow += layer_arr
    return np.clip(soft_shadow, 0.0, 1.0)


def project_shadow(
    mask_alpha: np.ndarray,
    angle_deg: float,
    elevation_deg: float,
    falloff: float,
    shadow_opacity: float,
    contact_strength: float,
    contact_threshold: float,
    place_x: int,
    place_y: int,
    bg_size: tuple[int, int],
    depth_map: np.ndarray | None,
    depth_strength: float,
) -> np.ndarray:
    bg_w, bg_h = bg_size
    h, w = mask_alpha.shape
    if mask_alpha.max() == 0:
        return np.zeros((bg_h, bg_w), dtype=np.float32)

    mask_canvas = np.zeros((bg_h, bg_w), dtype=np.uint8)
    height_canvas = np.zeros((bg_h, bg_w), dtype=np.uint8)
    y0 = place_y + h - 1
    x0 = place_x

    if y0 < 0 or x0 < 0 or x0 + w > bg_w or y0 - (h - 1) < 0:
        return np.zeros((bg_h, bg_w), dtype=np.float32)

    mask_canvas[place_y : place_y + h, place_x : place_x + w] = mask_alpha
    max_height = max(1, h - 1)
    heights = (h - 1 - np.arange(h, dtype=np.float32))[:, None]
    height_local = heights * (mask_alpha.astype(np.float32) / 255.0)
    height_scaled = np.clip(height_local / max_height * 255.0, 0, 255).astype(np.uint8)
    height_canvas[place_y : place_y + h, place_x : place_x + w] = height_scaled

    elev = math.radians(max(0.5, min(89.5, elevation_deg)))
    cot = 1.0 / math.tan(elev)
    shadow_angle = math.radians(angle_deg % 360.0)
    dx = math.cos(shadow_angle)
    dy = math.sin(shadow_angle)
    kx = dx * cot
    ky = dy * cot
    if abs(ky) < 1e-4:
        ky = 1e-4 if ky >= 0 else -1e-4

    a = 1.0
    b = -kx / ky
    c = (kx * y0) / ky
    d = 0.0
    e = -1.0 / ky
    f = (y0 / ky) + y0
    matrix = (a, b, c, d, e, f)

    mask_img = Image.fromarray(mask_canvas)
    height_img = Image.fromarray(height_canvas)
    shadow_mask = mask_img.transform((bg_w, bg_h), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=0)
    height_warp = height_img.transform((bg_w, bg_h), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=0)

    shadow_base = np.asarray(shadow_mask).astype(np.float32) / 255.0
    height_warped = np.asarray(height_warp).astype(np.float32) / 255.0 * max_height
    shadow_len = height_warped * cot
    max_shadow = max(bg_w, bg_h) * 1.5
    shadow_len = np.clip(shadow_len, 0.0, max_shadow)

    if depth_map is not None:
        depth_samples = depth_map.astype(np.float32)
        depth_offset = (depth_samples - 128.0) / 128.0 * depth_strength
        shift_x = np.round(depth_offset * dx).astype(np.int32)
        shift_y = np.round(depth_offset * dy).astype(np.int32)
        shadow_shifted = np.zeros_like(shadow_base)
        len_shifted = np.full_like(shadow_len, np.inf, dtype=np.float32)
        ys, xs = np.nonzero(shadow_base > 0)
        nx = np.clip(xs + shift_x[ys, xs], 0, bg_w - 1)
        ny = np.clip(ys + shift_y[ys, xs], 0, bg_h - 1)
        flat_idx = ny * bg_w + nx
        shadow_flat = shadow_shifted.ravel()
        len_flat = len_shifted.ravel()
        np.maximum.at(shadow_flat, flat_idx, shadow_base[ys, xs])
        np.minimum.at(len_flat, flat_idx, shadow_len[ys, xs])
        shadow_base = shadow_shifted
        shadow_len = len_shifted

    shadow_map = shadow_base * np.exp(-shadow_len / max(1.0, falloff))
    dist_map = np.where(shadow_base > 0, shadow_len, np.inf)

    bins = [(0.0, 40.0), (40.0, 90.0), (90.0, 160.0), (160.0, 300.0), (300.0, 1e9)]
    radii = [1.0, 3.0, 6.0, 10.0, 16.0]
    soft_shadow = blur_by_distance(shadow_map, dist_map, bins, radii) * shadow_opacity

    contact_band = (dist_map <= max(1.0, contact_threshold)).astype(np.float32)
    contact_layer = shadow_map * contact_band
    contact_img = Image.fromarray(np.clip(contact_layer * 255.0, 0, 255).astype(np.uint8))
    contact_img = contact_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    contact_shadow = np.asarray(contact_img).astype(np.float32) / 255.0

    final_shadow = np.clip(soft_shadow + contact_shadow * contact_strength, 0.0, 1.0)
    return final_shadow


def main() -> int:
    parser = argparse.ArgumentParser(description="Composite a subject with a realistic directional shadow.")
    parser.add_argument("--foreground", required=True, help="Path to the foreground image.")
    parser.add_argument("--background", required=True, help="Path to the background image.")
    parser.add_argument("--depth", default=None, help="Optional depth map (grayscale 0-255).")
    parser.add_argument("--angle", type=float, default=315.0, help="Light angle in degrees (0-360).")
    parser.add_argument("--elevation", type=float, default=35.0, help="Light elevation in degrees (0-90).")
    parser.add_argument("--scale", type=float, default=0.65, help="Scale factor for the foreground.")
    parser.add_argument("--x", type=int, default=None, help="Foreground x offset (top-left).")
    parser.add_argument("--y", type=int, default=None, help="Foreground y offset (top-left).")
    parser.add_argument("--mask-threshold", type=int, default=None, help="Override mask threshold (0-255).")
    parser.add_argument("--mask-blur", type=float, default=2.0, help="Mask edge blur radius.")
    parser.add_argument("--border-size", type=int, default=12, help="Border size for background sampling.")
    parser.add_argument("--falloff", type=float, default=180.0, help="Shadow opacity falloff distance.")
    parser.add_argument("--shadow-opacity", type=float, default=0.65, help="Global shadow opacity.")
    parser.add_argument("--contact-strength", type=float, default=0.9, help="Contact shadow strength.")
    parser.add_argument("--contact-threshold", type=float, default=10.0, help="Max height for contact shadow.")
    parser.add_argument("--depth-strength", type=float, default=30.0, help="Depth warp strength.")
    parser.add_argument("--output", default="composite.png", help="Composite output path.")
    parser.add_argument("--shadow-only", default="shadow_only.png", help="Shadow debug output path.")
    parser.add_argument("--mask-debug", default="mask_debug.png", help="Mask debug output path.")
    args = parser.parse_args()

    fg_raw = Image.open(args.foreground)
    fg = fg_raw.convert("RGB")
    bg = load_rgb(Path(args.background))

    if args.depth:
        depth = Image.open(args.depth).convert("L").resize(bg.size, resample=Image.BILINEAR)
        depth_map = np.asarray(depth)
    else:
        depth_map = None

    has_alpha = fg_raw.mode in ("RGBA", "LA") or (
        fg_raw.mode == "P" and "transparency" in fg_raw.info
    )
    if has_alpha:
        fg_rgba = fg_raw.convert("RGBA")
        mask_img = fg_rgba.split()[-1]
    else:
        mask_img = generate_mask(fg, args.mask_threshold, args.border_size, args.mask_blur)

    if args.scale != 1.0:
        new_size = (int(fg.width * args.scale), int(fg.height * args.scale))
        fg = fg.resize(new_size, resample=Image.LANCZOS)
        mask_img = mask_img.resize(new_size, resample=Image.LANCZOS)

    mask_img.save(args.mask_debug)

    if args.x is None:
        place_x = (bg.width - fg.width) // 2
    else:
        place_x = args.x
    if args.y is None:
        place_y = bg.height - fg.height
    else:
        place_y = args.y

    mask_alpha = np.asarray(mask_img).astype(np.uint8)
    shadow_alpha = project_shadow(
        mask_alpha,
        angle_deg=args.angle,
        elevation_deg=args.elevation,
        falloff=args.falloff,
        shadow_opacity=args.shadow_opacity,
        contact_strength=args.contact_strength,
        contact_threshold=args.contact_threshold,
        place_x=place_x,
        place_y=place_y,
        bg_size=(bg.width, bg.height),
        depth_map=depth_map,
        depth_strength=args.depth_strength,
    )

    shadow_alpha_img = Image.fromarray(np.clip(shadow_alpha * 255.0, 0, 255).astype(np.uint8))
    shadow_alpha_img.save(args.shadow_only)

    shadow_img = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    shadow_img.putalpha(shadow_alpha_img)

    bg_rgba = bg.convert("RGBA")
    comp = Image.alpha_composite(bg_rgba, shadow_img)

    fg_rgba = fg.convert("RGBA")
    fg_rgba.putalpha(mask_img)
    comp.paste(fg_rgba, (place_x, place_y), fg_rgba)
    comp.save(args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
