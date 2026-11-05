"""The pinned scan remains portable and byte-identical after XML deduplication."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.cld_ecal_readout.prepare_granularity_scan import ROOT, SOURCE, SOURCE_SHA256, main


def xml_bytes(directory):
    return {p.name: p.read_bytes() for p in directory.glob("*.xml")}


def test_relocated_source_reproduces_all_committed_variants(tmp_path):
    source = tmp_path / "relocated-k4geo"
    shutil.copytree(SOURCE, source)
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
    source = xml_bytes(SOURCE)
    assert hashlib.sha256(b"".join(name.encode() + source[name] for name in sorted(source))).hexdigest() == SOURCE_SHA256
    shared = SOURCE.resolve()
    for path in ROOT.parent.rglob("*.xml"):
        assert path.is_file()
        if path.is_relative_to(SOURCE):
            assert not path.is_symlink()
        elif path.name == "CLD_o2_v07.xml" or path.read_bytes() != source[path.name]:
            # Every changed file is local, even if another scenario has the same change.
            assert not path.is_symlink()
        else:
            assert path.is_symlink()
            assert not Path(path.readlink()).is_absolute()
            assert path.resolve() == shared / path.name


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


@pytest.mark.parametrize("mode", ["ecal", "hcal"])
def test_production_compacts_select_scan_pitch_and_step_limits(mode):
    from xml.etree import ElementTree as ET
    import yaml

    repository = Path(__file__).resolve().parents[1]
    productions = yaml.safe_load((repository / "particleflow_spec.yaml").read_text())["productions"]
    selected = {name: config for name, config in productions.items() if name.startswith(f"cld_{mode}_pitch_")}
    assert len(selected) == 6
    for name, config in selected.items():
        pitch = float(name.rsplit("_", 1)[1].replace("p", "."))
        compact = repository / config["compact_file"].replace("${project.paths.project_root}/", "")
        tree = ET.parse(compact)
        includes = {node.get("ref") for node in tree.findall("include")}
        subdetectors = {}
        for subsystem, version in (("ECal", "o2_v01_03"), ("HCal", "o1_v01_01")):
            for part in ("Barrel", "Endcap"):
                filename = f"{subsystem}{part}_{version}.xml"
                assert filename in includes
                subdetectors[subsystem, part] = ET.parse(compact.parent / filename)
        ecal_pitch = subdetectors["ECal", "Barrel"].find(".//constant[@name='ECal_cell_size']").get("value")
        assert ecal_pitch == (f"{pitch:.1f}*mm" if mode == "ecal" else "5.1*mm")
        for part in ("Barrel", "Endcap"):
            hcal_pitch = subdetectors["HCal", part].find(".//constant[@name='HCal_cell_size']").get("value")
            assert hcal_pitch == (f"{pitch / 10:.1f}*cm" if mode == "hcal" else "3.0*cm")
            segmentation = subdetectors["ECal", part].find(f".//readout[@name='ECal{part}Collection']/segmentation")
            assert segmentation.get("grid_size_x") == segmentation.get("grid_size_y") == "ECal_cell_size"
            sensitive = subdetectors["ECal", part].find(".//slice[@sensitive='yes']")
            assert sensitive.get("limits") == ("ecal_limits" if mode == "ecal" else "cal_limits")
        step = tree.find(f".//limitset[@name='{'ecal_limits' if mode == 'ecal' else 'cal_limits'}']/limit")
        assert step.get("value") == ("1.0" if mode == "ecal" else "5.0")


@pytest.mark.parametrize("production", ["cld_ecal_pitch_2", "cld_hcal_pitch_10"])
def test_scan_workflow_selects_runner_and_geometry(tmp_path, monkeypatch, production):
    import subprocess
    import sys
    import yaml
    from mlpf.snakemake import produce_snakemake as workflow

    repository = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((repository / "particleflow_spec.yaml").read_text())
    spec["project"]["paths"].update(
        project_root=str(repository), storage_root=str(tmp_path / "storage"), scratch_root=str(tmp_path / "scratch"), tmpdir=str(tmp_path / "tmp")
    )
    monkeypatch.setattr(workflow, "load_spec", lambda _: spec)
    monkeypatch.setattr(workflow, "LOCAL_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(sys, "argv", ["produce_snakemake.py", "--production", production])
    workflow.main()
    config = spec["productions"][production]
    sample_name, sample = next(iter(config["samples"].items()))
    runner = repository / sample["gen_script"]
    assert runner.is_file()
    directory = tmp_path / "jobs" / production
    generated = directory / "gen" / f"gen_{sample_name}.sh"
    script = generated.read_text()
    compact = config["compact_file"].replace("${project.paths.project_root}", str(repository))
    assert f'export CLD_COMPACT_FILE="{compact}"' in script
    assert f"export CLD_CMS_ENERGY={config['cms_energy']}" in script
    assert f"bash {sample['gen_script']} {sample['process_name']}" in script
    for path in (runner, generated):
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert {p.name for p in directory.glob("Snakefile_*")} == {"Snakefile_post", "Snakefile_tfds", "Snakefile_tfds_hit"}
