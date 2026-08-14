import json

import numpy as np

from scripts.local.plot_training_evolution import load_history, plot_training_evolution


def test_load_and_plot_training_evolution(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    metric_prefix = "step/cld_edm_ttbar_hits/jet_ratio/jet_ratio_target_to_pred_pt/"
    for step, loss in [(4000, 0.8), (2000, 1.0)]:
        data = {
            "train": {"Total": loss},
            "valid": {"Total": loss + 0.1},
            f"{metric_prefix}match_frac": 0.5 + step / 20_000,
            f"{metric_prefix}iqr": 0.5 - step / 20_000,
            f"{metric_prefix}med": 0.7 + step / 20_000,
        }
        (history_dir / f"step_{step}.json").write_text(json.dumps(data))

    history = load_history(history_dir, "cld_edm_ttbar_hits")

    np.testing.assert_array_equal(history["step"], [2000, 4000])
    np.testing.assert_allclose(history["train_loss"], [1.0, 0.8])
    np.testing.assert_allclose(history["valid_loss"], [1.1, 0.9])

    output_dir = tmp_path / "plots"
    plot_training_evolution(history, output_dir)
    assert (output_dir / "training_evolution.png").stat().st_size > 0
    assert (output_dir / "training_evolution.pdf").stat().st_size > 0
