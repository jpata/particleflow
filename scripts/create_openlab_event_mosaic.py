#!/usr/bin/env python3
"""Create a text-free, wide event artwork inspired by the 2025 openlab report.

The source displays are real transverse event plots from
run_detector_event_displays.sh. Columns are ordered ColliderML, CLIC, CLD,
IDEA; the default upper/lower rows are independent event indices 0 and 1.
Each detector column uses a fixed crop across rows. Crops differ by detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DETECTORS = ("colliderml", "clic", "cld", "idea")
HALF_WIDTH = {"colliderml": 310, "clic": 290, "cld": 365, "idea": 395}
NAVY = (5, 15, 36)
COLUMN_TINTS = ((13, 8, 12), (4, 16, 15), (7, 11, 24), (13, 8, 23))
ACCENTS = ((219, 145, 239), (64, 221, 218), (91, 166, 255), (175, 146, 255))


def detector_silhouettes(size: tuple[int, int]) -> Image.Image:
    """Faint abstract envelopes that repeat within each detector column."""
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    radius_sets = ((116, 188, 250), (165, 260), (125, 225, 286), (282, 315))
    for col, (radii, accent) in enumerate(zip(radius_sets, ACCENTS)):
        cx = 400 + col * 785
        for cy in (416, 1164):
            for index, radius in enumerate(radii):
                color = accent + (19 if index < len(radii) - 1 else 29,)
                box = (cx - radius, cy - radius, cx + radius, cy + radius)
                # Broken arcs read as atmosphere, not exact detector geometry.
                if col == 0:
                    draw.arc(box, 18, 158, fill=color, width=2)
                    draw.arc(box, 201, 339, fill=color, width=2)
                elif col == 1:
                    draw.arc(box, 34, 146, fill=color, width=2)
                    draw.arc(box, 217, 326, fill=color, width=2)
                elif col == 2:
                    draw.arc(box, 8, 174, fill=color, width=2)
                    draw.arc(box, 190, 350, fill=color, width=2)
                else:
                    draw.arc(box, 0, 359, fill=color, width=2)
    return overlay


def event_layer(source: Path, detector: str, side: int) -> Image.Image:
    """Map white-backed plot marks to a transparent luminous palette."""
    with Image.open(source) as original:
        scale_x = original.width / 1157
        scale_y = original.height / 1193
        if abs(scale_x - scale_y) > 0.03:
            raise ValueError(f"unexpected source aspect ratio for {source}: {original.size}")
        radius = HALF_WIDTH[detector]
        crop = original.convert("RGB").crop(
            (round((580 - radius) * scale_x), round((610 - radius) * scale_y),
             round((580 + radius) * scale_x), round((610 + radius) * scale_y))
        )
    rgb = np.asarray(crop.resize((side, side), Image.LANCZOS), dtype=np.float32)
    low = rgb.min(axis=2)
    high = rgb.max(axis=2)
    contrast = np.clip((255.0 - low) / 170.0, 0, 1)
    saturation = high - low
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # White paper vanishes; dark reconstructed tracks become icy white.
    # Detector hit hues are retained in a smaller, brighter openlab palette.
    out = np.empty_like(rgb)
    out[:] = (226, 239, 255)
    red = (r > g * 1.20) & (r > b * 1.13) & (saturation > 32)
    green = (g > r * 1.14) & (g > b * 1.08) & (saturation > 32)
    blue = (b > r * 1.16) & (b > g * 1.08) & (saturation > 28)
    violet = (r > g * 1.17) & (b > g * 1.17) & (saturation > 25)
    amber = red & (g > b * 1.25)
    out[red] = (255, 123, 105)
    out[green] = (60, 235, 208)
    out[blue] = (55, 183, 255)
    out[violet] = (191, 138, 255)
    out[amber] = (255, 177, 79)
    alpha = np.clip(contrast**0.72 * 245, 0, 245)
    alpha[low > 252] = 0
    rgba = np.dstack((out.astype(np.uint8), alpha.astype(np.uint8)))
    return Image.fromarray(rgba, "RGBA")


def add_glow(canvas: Image.Image, layer: Image.Image, x: int, y: int, scale: int) -> None:
    glow = layer.copy()
    glow.putalpha(glow.getchannel("A").point(lambda a: int(a * 0.18)))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18 * scale)), (x, y))
    canvas.alpha_composite(layer, (x, y))


def make_figure(input_dir: Path, output: Path, events: tuple[int, ...], scale: int = 1) -> None:
    if len(events) != 2:
        raise ValueError("the wide artwork requires exactly two event indices")
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    width, height = 3200, 1580
    # Diffuse navy-blue illumination follows the report's wide image language.
    # The gradients are continuous, avoiding artificial detector outlines.
    yy, xx = np.ogrid[:height, :width]
    background = np.empty((height, width, 3), dtype=np.float32)
    background[:] = NAVY
    for col in range(4):
        cx = 400 + col * 785
        tint = np.array(COLUMN_TINTS[col], dtype=np.float32)
        column_field = np.exp(-((xx - cx) ** 2 / (2 * 330**2)))
        background += column_field[:, :, None] * tint * 0.65
        for cy in (410, 1160):
            field = np.exp(-((xx - cx) ** 2 / (2 * 540**2) + (yy - cy) ** 2 / (2 * 420**2)))
            background += field[:, :, None] * tint
    background = np.clip(background, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(background, "RGB").convert("RGBA")
    canvas = Image.alpha_composite(canvas, detector_silhouettes(canvas.size))

    # These hairlines group the two rows in each column without boxing them in.
    boundaries = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    boundary_draw = ImageDraw.Draw(boundaries, "RGBA")
    for x in (792, 1577, 2362):
        boundary_draw.line((x, 95, x, height - 95), fill=(84, 157, 235, 24), width=2)
    canvas = Image.alpha_composite(canvas, boundaries)
    if scale > 1:
        canvas = canvas.resize((width * scale, height * scale), Image.LANCZOS)

    side = 700 * scale
    for row, event in enumerate(events):
        y = (66 + row * 748) * scale
        for col, detector in enumerate(DETECTORS):
            source = input_dir / f"{detector}_event_{event}.png"
            if not source.is_file():
                raise FileNotFoundError(f"missing source display: {source}")
            layer = event_layer(source, detector, side)
            x = (50 + col * 785) * scale
            add_glow(canvas, layer, x, y, scale)

    # A restrained scientific network texture sits between the event rows.
    # It is decorative and deliberately faint enough not to resemble hits.
    surface = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    net = ImageDraw.Draw(surface, "RGBA")
    anchors = [(90, 795), (585, 770), (1030, 795), (1540, 766), (2050, 805), (2580, 774), (3120, 795)]
    for a, b in zip(anchors, anchors[1:]):
        net.line((a, b), fill=(25, 127, 215, 26), width=2)
    for x, y in anchors:
        net.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(49, 184, 255, 50))
    if scale > 1:
        surface = surface.resize(canvas.size, Image.LANCZOS)
    canvas = Image.alpha_composite(canvas, surface)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, optimize=True, dpi=(150 * scale, 150 * scale))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("local_test_data/detector_comparison/event_displays"))
    parser.add_argument("--output", type=Path, default=Path("local_test_data/detector_comparison/event_displays/openlab_multi_event_dark.png"))
    parser.add_argument("--events", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--scale", type=int, default=1, help="output pixel scale (default: 1)")
    args = parser.parse_args()
    make_figure(args.input_dir, args.output, tuple(args.events), scale=args.scale)
    print(args.output)


if __name__ == "__main__":
    main()
