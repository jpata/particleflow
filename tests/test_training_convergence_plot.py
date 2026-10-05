import json
import sys

from torch.utils.tensorboard import SummaryWriter

from scripts.plot_training_convergence import LOSSES, main


def test_convergence_plot_preserves_validation_and_relative_change(tmp_path, monkeypatch):
    writer = SummaryWriter(str(tmp_path / "runs/train"))
    for step in (25, 50, 100, 125, 150, 175, 200, 225, 250):
        for name in LOSSES:
            writer.add_scalar("step/loss_" + name, 10 / step, step)
    writer.close()
    (tmp_path / "history").mkdir()
    for step, loss in ((225, 2.0), (450, 1.0)):
        (tmp_path / "history" / f"step_{step}.json").write_text(json.dumps({"valid": {name: loss for name in LOSSES}}))
    (tmp_path / "hyperparameters.json").write_text(json.dumps({"task_loss_weights": {"calibration_steps": 100}}))
    monkeypatch.setattr(sys, "argv", ["plot_training_convergence.py", "--experiment-dir", str(tmp_path)])
    main()
    summary = json.loads((tmp_path / "convergence.json").read_text())
    assert [point["step"] for point in summary["validation"]] == [225, 450]
    assert summary["validation_total_relative_change"] == -0.5
    assert (tmp_path / "convergence.png").stat().st_size > 0
