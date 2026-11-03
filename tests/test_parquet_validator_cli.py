"""The validator has one CLI workflow and descriptive report names."""

import json
import sys

import awkward as ak
import pytest

from tests.validate_parquet import main


def test_help_lists_only_supported_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["validate_parquet.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in ["--input", "--detector", "--max-events", "--plots-dir"]:
        assert option in help_text
    assert "--mode" not in help_text
    assert "--alignment-only" not in help_text


@pytest.mark.parametrize("extra", [["--mode", "report"], ["--alignment-only"], ["--max-events", "0"]])
def test_invalid_options_are_rejected(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["validate_parquet.py", "--input", "unused.parquet", "--detector", "colliderml", *extra])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_missing_fields_produce_report_and_failure_exit(tmp_path, monkeypatch):
    source, output = tmp_path / "input.parquet", tmp_path / "plots"
    ak.to_parquet(ak.Array({"genmet": [1.0]}), source)
    monkeypatch.setattr(sys, "argv", ["validate_parquet.py", "--input", str(source), "--detector", "colliderml", "--plots-dir", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads((output / "validation_report.json").read_text())
    checks = {g["gate_id"]: g for g in report["gates"]}
    assert checks["file_contents"]["status"] == "PASS"
    assert checks["required_fields"]["status"] == "FAIL"
    assert (output / "required_fields.png").exists()
    assert not (output / "gate_S2.png").exists()


@pytest.mark.parametrize("passed,exit_code", [(True, 0), (False, 1)])
def test_directory_uses_same_checks_for_each_file(tmp_path, monkeypatch, passed, exit_code):
    from tests.validate_parquet import ParquetValidator

    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    for name in ["one.parquet", "two.parquet"]:
        (source / name).touch()
    calls = []

    def run(validator):
        calls.append((validator.input_path.name, validator.plots_dir))
        return passed

    monkeypatch.setattr(ParquetValidator, "run", run)
    monkeypatch.setattr(sys, "argv", ["validate_parquet.py", "--input", str(source), "--detector", "colliderml", "--plots-dir", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == exit_code
    assert calls == [("one.parquet", output / "one"), ("two.parquet", output / "two")]
