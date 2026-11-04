"""Build reproducible CLD ECAL/HCAL pitch variants from pinned k4geo XML.

Run with ``--source`` pointing to the 2025-05-29 k4geo
FCCee/CLD/compact/CLD_o2_v07 directory. Existing variants are verified and
never silently overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path


ECAL_MM = (5.1, 4.0, 3.0, 2.5, 2.0, 1.5)
HCAL_MM = (30.0, 25.0, 20.0, 15.0, 12.0, 10.0)
ROOT = Path(__file__).resolve().parents[2] / "mlpf/data/key4hep/geometry/granularity_scan"


def label(value):
    return f"{value:g}".replace(".", "p")


def replace_one(contents, old, new, filename):
    count = contents.count(old)
    if count != 1:
        raise ValueError(f"{filename}: expected one occurrence of {old!r}; found {count}")
    return contents.replace(old, new)


def variants(source, mode, pitch):
    result = {path.name: path.read_bytes() for path in source.glob("*.xml")}
    if mode == "ecal":
        name = "ECalBarrel_o2_v01_03.xml"
        content = result[name].decode()
        content = replace_one(content, 'name="ECal_cell_size" value="5.1*mm"', f'name="ECal_cell_size" value="{pitch:.1f}*mm"', name)
        content = replace_one(content, 'sensitive="yes" limits="cal_limits"', 'sensitive="yes" limits="ecal_limits"', name)
        result[name] = content.encode()
        name = "ECalEndcap_o2_v01_03.xml"
        content = result[name].decode()
        content = replace_one(content, 'sensitive="yes" limits="cal_limits"', 'sensitive="yes" limits="ecal_limits"', name)
        result[name] = content.encode()
        name = "CLD_o2_v07.xml"
        content = result[name].decode()
        old = '<limitset name="cal_limits">\n            <limit name="step_length_max" particles="*" value="5.0" unit="mm" />\n        </limitset>'
        new = (
            old
            + '\n        <limitset name="ecal_limits">\n            <limit name="step_length_max" particles="*" value="1.0" unit="mm" />\n        </limitset>'
        )
        content = replace_one(content, old, new, name)
        result[name] = content.encode()
    elif mode == "hcal":
        for name in ("HCalBarrel_o1_v01_01.xml", "HCalEndcap_o1_v01_01.xml"):
            content = result[name].decode()
            content = replace_one(content, 'name="HCal_cell_size" value="3.0*cm"', f'name="HCal_cell_size" value="{pitch / 10:.1f}*cm"', name)
            result[name] = content.encode()
    else:
        raise ValueError(mode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "CLD_o2_v07.xml").is_file():
        raise FileNotFoundError(source / "CLD_o2_v07.xml")
    source_files = {p.name: p.read_bytes() for p in source.glob("*.xml")}
    if len(source_files) != 22:
        raise ValueError(f"Expected 22 source XML files; found {len(source_files)}")
    source_digest = hashlib.sha256(b"".join(name.encode() + source_files[name] for name in sorted(source_files))).hexdigest()
    for mode, pitches in (("ecal", ECAL_MM), ("hcal", HCAL_MM)):
        for pitch in pitches:
            directory = args.output_root / f"{mode}_{label(pitch)}mm"
            expected = variants(source, mode, pitch)
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in expected.items():
                dest = directory / name
                if dest.exists():
                    if dest.read_bytes() != content:
                        raise ValueError(f"Existing geometry differs: {dest}")
                else:
                    dest.write_bytes(content)
            extra = {p.name for p in directory.glob("*.xml")} - expected.keys()
            if extra:
                raise ValueError(f"Unexpected files in {directory}: {sorted(extra)}")
            changed = sorted(name for name in expected if expected[name] != source_files[name])
            manifest = {
                "source": str(source),
                "source_sha256": source_digest,
                "mode": mode,
                "pitch_mm": pitch,
                "changed_files": changed,
                "ecal_step_limit_mm": 1.0 if mode == "ecal" else 5.0,
                "compact_file": "CLD_o2_v07.xml",
            }
            manifest_path = directory / "manifest.json"
            rendered = json.dumps(manifest, indent=2) + "\n"
            if manifest_path.exists() and manifest_path.read_text() != rendered:
                raise ValueError(f"Existing manifest differs: {manifest_path}")
            manifest_path.write_text(rendered)
            print(f"{directory}: {', '.join(changed)}")


if __name__ == "__main__":
    main()
