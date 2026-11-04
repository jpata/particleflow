"""The pinned scan remains portable and byte-identical after XML deduplication."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.cld_ecal_readout.prepare_granularity_scan import ROOT, SOURCE_SHA256, main


def xml_bytes(directory):
    return {p.name: p.read_bytes() for p in directory.glob("*.xml")}


def test_relocated_source_reproduces_all_committed_variants(tmp_path):
    source = tmp_path / "relocated-k4geo"
    shutil.copytree(ROOT / "hcal_30mm", source)
    output = tmp_path / "generated"
    main(["--source", str(source), "--output-root", str(output)])
    for committed in sorted(ROOT.iterdir()):
        if not committed.is_dir():
            continue
        generated = output / committed.name
        assert xml_bytes(generated) == xml_bytes(committed)
        assert not (committed / "CLD_o2_v07.xml").is_symlink()
        assert all(not p.is_symlink() for p in generated.glob("*.xml"))
    # A second location with identical bytes must verify without changing provenance.
    moved = tmp_path / "moved-k4geo"
    shutil.copytree(source, moved)
    before = {p: p.read_bytes() for p in output.glob("*/manifest.json")}
    main(["--source", str(moved), "--output-root", str(output)])
    assert before == {p: p.read_bytes() for p in before}
    assert len(before) == 12


def test_committed_source_and_relative_links_stay_inside_geometry_tree():
    source = xml_bytes(ROOT / "hcal_30mm")
    assert hashlib.sha256(b"".join(name.encode() + source[name] for name in sorted(source))).hexdigest() == SOURCE_SHA256
    geometry = ROOT.parent.resolve()
    for path in geometry.rglob("*.xml"):
        assert path.is_file()
        if path.is_symlink():
            assert not Path(path.readlink()).is_absolute()
            assert path.resolve().is_relative_to(geometry)


@pytest.mark.parametrize("damage", ["source", "variant", "extra", "manifest"])
def test_scan_verification_rejects_changed_content(tmp_path, damage):
    output = tmp_path / "generated"
    main(["--output-root", str(output)])
    source = output / "hcal_30mm"
    if damage == "source":
        path = source / "materials.xml"
        path.write_text(path.read_text() + "\n")
        expected = "does not match pinned"
    elif damage == "variant":
        path = output / "ecal_2mm/ECalBarrel_o2_v01_03.xml"
        path.write_text(path.read_text().replace("2.0*mm", "2.1*mm"))
        expected = "Existing geometry differs"
    elif damage == "extra":
        (output / "ecal_2mm/unexpected.xml").write_text("<lccdd/>")
        expected = "Unexpected files"
    else:
        path = output / "ecal_2mm/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pitch_mm"] = 2.1
        path.write_text(json.dumps(manifest))
        expected = "Existing manifest differs"
    with pytest.raises(ValueError, match=expected):
        main(["--source", str(source), "--output-root", str(output)])
