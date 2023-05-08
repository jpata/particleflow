import os
from pathlib import Path

import mplhep
from plotting.plot_utils import (
    compute_met_and_ratio,
    format_dataset_name,
    load_eval_data,
    plot_jet_ratio,
    plot_met,
    plot_met_ratio,
    plot_num_elements,
    plot_particles,
    plot_sum_energy,
)


def make_plots(pred_path, plot_path, dataset, sample):
    mplhep.set_style(mplhep.styles.CMS)

    pred_path = Path(pred_path)
    plot_path = Path(plot_path)

    # Use the dataset names from the common nomenclature
    _title = "NA"
    if dataset == "CLIC":
        if sample == "QCD":
            _title = format_dataset_name("clic_edm_qq_pf")
        elif sample == "TTbar":
            _title = format_dataset_name("clic_edm_ttbar_pf")

    if not os.path.isdir(str(plot_path)):
        os.makedirs(str(plot_path))
    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), -1)

    plot_num_elements(X, cp_dir=plot_path, title=_title)
    plot_sum_energy(yvals, cp_dir=plot_path, title=_title)

    plot_jet_ratio(yvals, cp_dir=plot_path, title=_title)

    met_data = compute_met_and_ratio(yvals)
    plot_met(met_data, cp_dir=plot_path, title=_title)
    plot_met_ratio(met_data, cp_dir=plot_path, title=_title)

    plot_particles(yvals, cp_dir=plot_path, title=_title)
