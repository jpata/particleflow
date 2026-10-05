import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scripts.compare_detector_features import DETECTORS, layout_comparison_figure


@pytest.mark.parametrize("rows", [1, 2, 5, 6])
@pytest.mark.parametrize("synthetic", [False, True])
def test_header_titles_do_not_overlap_each_other_or_panels(rows, synthetic):
    fig, axes = plt.subplots(rows, len(DETECTORS), figsize=(20, max(3.6, 2.35 * rows)), squeeze=False)
    for ax in axes.flat:
        ax.set_title("Individual plot title")
        ax.set_xlabel("Feature label")
    main, columns = layout_comparison_figure(
        fig, axes, "ttbar • parquet • tracks — shared bins/axes; union of detector 0.5–99.5% ranges", synthetic)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    main_box = main.get_window_extent(renderer)
    for column, ax in zip(columns, axes[0]):
        box = column.get_window_extent(renderer)
        assert main_box.y0 > box.y1
        assert box.y0 > ax.get_tightbbox(renderer).y1
    plt.close(fig)
