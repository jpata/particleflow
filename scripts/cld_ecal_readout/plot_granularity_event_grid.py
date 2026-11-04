"""Render one paired CLD event at all six pitches and assemble a grid.

Uses the existing scripts/visualize_key4hep.py renderer with the same camera
and hit limit for every point. Run separately for ecal and hcal scans.
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize_key4hep import render_event

BASE = Path("/local/joosep/mlpf/cld/granularity_scan_2025-05-29")
PITCHES = {"ecal": ["5p1", "4", "3", "2p5", "2", "1p5"], "hcal": ["30", "25", "20", "15", "12", "10"]}
SAMPLE = {"ecal": "p8_ee_Z_tautau_ecm91", "hcal": "p8_ee_ttbar_ecm365"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=PITCHES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument(
        "--zoom-xy", type=float, nargs=2, metavar=("X", "Y"), help="also render a fixed transverse x-y region in display coordinates [mm]"
    )
    parser.add_argument("--zoom-limit", type=float, default=220, help="half-width of the optional zoom view [mm]")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cld_granularity_scan/events"))
    parser.add_argument("--max-hits", type=int, default=1000000, help="cap per hit collection; the default retains all hits in these events")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (700000 if args.group == "ecal" else 300000)
    if args.group == "ecal" and not 700000 <= seed <= 700099:
        parser.error("ECAL scan seeds are 700000–700099")
    if args.group == "hcal" and not 300000 <= seed <= 300099:
        parser.error("HCAL scan seeds are 300000–300099")
    out = args.output_dir / f"{args.group}_seed{seed}_event{args.event}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    truth = None
    sample = SAMPLE[args.group]
    for pitch in PITCHES[args.group]:
        point = f"{args.group}_{pitch}mm"
        root = BASE / point / "gen" / sample / "root" / f"reco_{sample}_{seed}.root"
        if not root.is_file():
            raise FileNotFoundError(root)
        with uproot.open(root, handler=uproot.source.file.MemmapSource) as f:
            tree = f["events"]
            if not 0 <= args.event < tree.num_entries:
                raise ValueError(f"{root}: event {args.event} outside file")
            pdg = ak.to_list(tree["MCParticles.PDG"].array(entry_start=args.event, entry_stop=args.event + 1)[0])
            status = ak.to_list(tree["MCParticles.generatorStatus"].array(entry_start=args.event, entry_stop=args.event + 1)[0])
            status1 = [p for p, s in zip(pdg, status) if s == 1]
            if truth is None:
                truth = status1
            elif status1 != truth:
                raise ValueError(f"Generator status-1 PDG sequence differs at {point}")
            views = ("combined", "hits", "zoom") if args.zoom_xy else ("combined", "hits")
            for view in views:
                image = out / f"{point}_{view}.png"
                render_event(
                    root,
                    args.event,
                    image,
                    max_hits=args.max_hits,
                    detector="cld",
                    plot_limit=args.zoom_limit if view == "zoom" else (2700 if args.group == "ecal" else 4200),
                    show_particles=view != "hits",
                    input_view="combined" if view == "zoom" else view,
                    tree=tree,
                    dpi=130,
                    center_xy=tuple(args.zoom_xy) if view == "zoom" else None,
                )
                paths.append((pitch, view, image))
                print(image, flush=True)
    for view in views:
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        for ax, pitch in zip(axs.flat, PITCHES[args.group]):
            image = next(path for p, v, path in paths if p == pitch and v == view)
            ax.imshow(plt.imread(image))
            ax.set_title(f"{pitch.replace('p', '.')} mm", fontsize=16)
            ax.axis("off")
        process = "Z→ττ" if args.group == "ecal" else "ttbar"
        fig.suptitle(f"CLD {args.group.upper()} pitch scan · {process} · seed {seed}, event {args.event} · {view}", fontsize=18)
        target = out / f"{args.group}_{view}_grid.png"
        fig.savefig(target, dpi=130, facecolor="white")
        plt.close(fig)
        print(target, flush=True)


if __name__ == "__main__":
    main()
