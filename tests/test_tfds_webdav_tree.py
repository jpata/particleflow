from pathlib import Path, PurePosixPath

import pytest


@pytest.fixture
def tree(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import tfds_webdav_tree

    return tfds_webdav_tree


def test_manifest_selects_only_requested_datasets(tree):
    remote = "davs://example.org/store/campaign/"
    listing = "\n".join(
        f"-rw-r--r-- 1 {size} Jan 1 /store/campaign/{path}"
        for size, path in [
            (10, "cld_edm_ttbar_hits/1/3.2.2/dataset_info.json"),
            (20, "cld_edm_qq_hits/1/3.2.2/dataset_info.json"),
            (30, "cld_edm_ttbar_pf/1/3.3.0/dataset_info.json"),
        ]
    )

    expected = tree.manifest(listing, remote, ["cld_edm_ttbar_hits", "cld_edm_qq_hits"])

    assert expected == {
        PurePosixPath("cld_edm_ttbar_hits/1/3.2.2/dataset_info.json"): 10,
        PurePosixPath("cld_edm_qq_hits/1/3.2.2/dataset_info.json"): 20,
    }
    with pytest.raises(ValueError, match="cld_edm_ww_fullhad_hits"):
        tree.manifest(listing, remote, ["cld_edm_ww_fullhad_hits"])


def test_local_progress_ignores_unselected_files(tree, tmp_path):
    selected = PurePosixPath("cld_edm_ttbar_hits/1/3.2.2/dataset_info.json")
    selected_file = tmp_path.joinpath(*selected.parts)
    selected_file.parent.mkdir(parents=True)
    selected_file.write_bytes(b"hits")
    other = tmp_path / "cld_edm_ttbar_pf" / "1" / "3.3.0"
    other.mkdir(parents=True)
    (other / "dataset_info.json").write_bytes(b"pf")

    assert tree.local_progress(tmp_path, {selected: 4}) == (1, 4)
