#!/usr/bin/env python3
"""Assemble shared-scale event displays into a detector comparison gallery."""
import argparse
from pathlib import Path

from scripts.create_openlab_event_mosaic import make_figure
from scripts.visualize_key4hep import render_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--events", type=int, nargs="+", required=True)
    args = parser.parse_args()
    detectors = ("colliderml", "clic", "cld", "idea", "maia")
    html = "<html><meta charset='utf-8'><title>ttbar event displays</title><style>body{font-family:sans-serif;margin:24px}img{width:100%}</style>"
    html += "<h1>ttbar event displays</h1><p>Independent collisions, not event-matched samples. Shared transverse scale: ±6500 mm; +x left, +y up, +z toward the viewer. Hits are sampled deterministically, up to 800 per collection/region; displayed point density is not an occupancy measurement. Cluster marker sizes follow the shared renderer's energy-based convention. IDEA tracks are truth-seeded. ColliderML clusters come from the converted parquet; track helices use ACTS parameters and nominal 3 T. MAIA helices use stored ROOT track states. Display trajectory envelopes are approximate, not detailed detector geometry.</p>"
    html += "<p>Dashed colored particle guides are visible status-1 truth for Key4hep/MAIA and primary leaves for ColliderML. Charged guides follow helices in each nominal axial field (CLIC 4 T; CLD/IDEA 2 T; ColliderML 3 T; MAIA 5 T); neutral guides remain straight. These are origin-based illustrative trajectories, without material interactions or energy loss; low-pT curling guides stop after at most one revolution. They are proxies, not the final merged/allocated MLPF targets. Neutrinos are omitted. Individual detector PNGs are also available below.</p>"
    if len(args.events) >= 2:
        mosaic = args.input / "openlab_multi_event_dark.png"
        make_figure(args.input, mosaic, tuple(args.events[:2]))
        html += "<h2>Four-detector, multi-event project artwork</h2><p>Text-free, dark-background view of ColliderML, CLIC, CLD and IDEA, in that left-to-right order. The upper and lower rows show the first two requested independent event indices. Each detector column keeps a fixed zoom across rows; scales differ between columns.</p>"
        html += f"<p><a href='{mosaic.name}'>Open PNG</a>"
        if tuple(args.events[:2]) == (0, 1):
            hires = args.input / "openlab_multi_event_dark_2x.png"
            if hires.is_file():
                html += f" · <a href='{hires.name}'>Open high-resolution PNG</a>"
        html += f"</p><img src='{mosaic.name}' alt='Two luminous event rows across ColliderML, CLIC, CLD and IDEA'>"
    for event in args.events:
        images = [(args.input / f"{detector}_event_{event}.png", detector) for detector in detectors]
        output = args.input / f"comparison_event_{event}.png"
        render_comparison(images, event, output)
        html += f"<h2>Event index {event}</h2><img src='{output.name}'>"
        html += "<p>" + " · ".join(f"<a href='{path.name}'>{detector.upper()}</a>" for path, detector in images) + "</p>"
    (args.input / "index.html").write_text(html + "</html>")
    print(args.input / "index.html")


if __name__ == "__main__":
    main()
