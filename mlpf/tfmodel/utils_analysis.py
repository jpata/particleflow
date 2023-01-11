from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def get_hp_str(result):
    def func(key):
        if "config" in key:
            return key.split("config/")[-1]

    s = ""
    for ii, hp in enumerate(list(filter(None.__ne__, [func(key) for key in result.keys()]))):
        if ii % 6 == 0:
            s += "\n"
        s += "{}={}; ".format(hp, result["config/{}".format(hp)].values[0])
    return s


def plot_ray_analysis(analysis, save=False, skip=0):
    to_plot = [
        "adam_beta_1",
        "charge_loss",
        "cls_acc_unweighted",
        "cls_loss",
        "cos_phi_loss",
        "energy_loss",
        "eta_loss",
        "learning_rate",
        "loss",
        "pt_loss",
        "sin_phi_loss",
        "val_charge_loss",
        "val_cls_acc_unweighted",
        "val_cls_acc_weighted",
        "val_cls_loss",
        "val_cos_phi_loss",
        "val_energy_loss",
        "val_eta_loss",
        "val_loss",
        "val_pt_loss",
        "val_sin_phi_loss",
    ]

    dfs = analysis.fetch_trial_dataframes()
    result_df = analysis.dataframe()
    for key in tqdm(dfs.keys(), desc="Creating Ray analysis plots", total=len(dfs.keys())):
        result = result_df[result_df["logdir"] == key]

        fig, axs = plt.subplots(5, 4, figsize=(12, 9), tight_layout=True)
        for var, ax in zip(to_plot, axs.flat):
            # Skip first `skip` values so loss plots don't include the very large losses which occur at start of training
            ax.plot(dfs[key].index.values[skip:], dfs[key][var][skip:], alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(var)
            ax.grid(alpha=0.3)
        plt.suptitle(get_hp_str(result))

        if save:
            plt.savefig(key + "/trial_summary.jpg")
            plt.close()
    if not save:
        plt.show()
    else:
        print("Saved plots in trial dirs.")


def correct_column_names_in_trial_dataframes(analysis):
    """
    Sometimes some trial dataframes are missing column names and have been
    given the first row of values as column names. This function corrects
    this in the ray.tune.Analysis object.
    """
    trial_dataframes = analysis.trial_dataframes
    trial_df_columns = [
        "adam_beta_1",
        "charge_loss",
        "cls_acc_unweighted",
        "cls_loss",
        "cos_phi_loss",
        "energy_loss",
        "eta_loss",
        "learning_rate",
        "loss",
        "pt_loss",
        "sin_phi_loss",
        "val_charge_loss",
        "val_cls_acc_unweighted",
        "val_cls_acc_weighted",
        "val_cls_loss",
        "val_cos_phi_loss",
        "val_energy_loss",
        "val_eta_loss",
        "val_loss",
        "val_pt_loss",
        "val_sin_phi_loss",
        "time_this_iter_s",
        "should_checkpoint",
        "done",
        "timesteps_total",
        "episodes_total",
        "training_iteration",
        "experiment_id",
        "date",
        "timestamp",
        "time_total_s",
        "pid",
        "hostname",
        "node_ip",
        "time_since_restore",
        "timesteps_since_restore",
        "iterations_since_restore",
        "trial_id",
    ]

    for ii, key in enumerate(trial_dataframes.keys()):
        trial_dataframes[key].columns = trial_df_columns

    analysis._trial_dataframes = trial_dataframes


def get_top_k_df(analysis, k):
    result_df = analysis.dataframe()
    if analysis.default_mode == "min":
        dd = result_df.nsmallest(k, analysis.default_metric)
    elif analysis.default_mode == "max":
        dd = result_df.nlargest(k, analysis.default_metric)
    return dd


def topk_summary_plot(analysis, k, save=False, save_dir=None):
    to_plot = [
        "val_cls_acc_unweighted",
        "val_cls_acc_weighted",
        "val_cls_loss",
        "val_energy_loss",
        "val_loss",
    ]

    dd = get_top_k_df(analysis, k)
    dfs = analysis.trial_dataframes

    fig, axs = plt.subplots(k, 5, figsize=(12, 9), tight_layout=True)
    for key, ax_row in zip(dd["logdir"], axs):

        for var, ax in zip(to_plot, ax_row):
            ax.plot(dfs[key].index.values, dfs[key][var], alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(var)
            ax.grid(alpha=0.3)

    if save:
        if save_dir:
            plt.savefig(str(Path(save_dir) / "topk_summary_plot.jpg"))
        else:
            plt.savefig("topk_summary_plot.jpg")
    else:
        plt.show()


def topk_summary_plot_v2(analysis, k, save=False, save_dir=None):
    print("Creating summary plot of top {} trials.".format(k))
    to_plot = [
        "val_loss",
        "val_cls_loss",
        "val_cls_acc_unweighted",
        "val_cls_acc_weighted",
    ]

    dd = get_top_k_df(analysis, k)
    dfs = analysis.trial_dataframes

    fig, axs = plt.subplots(len(to_plot), 1, figsize=(12, 9), tight_layout=True, sharex=True)
    for var, ax_row in zip(to_plot, axs):
        for ii, key in enumerate(dd["logdir"]):
            ax_row.plot(
                dfs[key].index.values,
                dfs[key][var],
                alpha=0.8,
                label="#{}".format(ii + 1),
            )
            ax_row.set_ylabel(var)
            ax_row.grid(alpha=0.3)
            ax_row.legend()
    ax_row.set_xlabel("Epoch")

    plt.suptitle("Top {} best trials according to '{}'".format(k, analysis.default_metric))
    if save or save_dir:
        if save_dir:
            file_name = str(Path(save_dir) / "topk_summary_plot_v2.jpg")
        else:
            file_name = "topk_summary_plot.jpg"
        plt.savefig(file_name)
        print("Saved summary plot to {}".format(file_name))
    else:
        plt.show()


def summarize_top_k(analysis, k, save=False, save_dir=None):
    print("Creating summary table of top {} trials.".format(k))
    dd = get_top_k_df(analysis, k)
    summary = pd.concat(
        [
            dd[
                [
                    "loss",
                    "cls_loss",
                    "cls_acc_unweighted",
                    "val_loss",
                    "val_cls_loss",
                    "val_cls_acc_unweighted",
                    "val_cls_acc_weighted",
                ]
            ],
            dd.filter(regex=("config/*")),
            dd["logdir"],
        ],
        axis=1,
    )
    cm_green = sns.light_palette("green", as_cmap=True)
    cm_red = sns.light_palette("red", as_cmap=True)

    max_is_better = [
        "cls_acc_unweighted",
        "val_cls_acc_weighted",
        "val_cls_acc_unweighted",
    ]
    min_is_better = ["loss", "cls_loss", "val_loss", "val_cls_loss"]

    styled_summary = (
        summary.style.background_gradient(cmap=cm_green, subset=max_is_better)
        .background_gradient(cmap=cm_red, subset=min_is_better)
        .highlight_max(
            subset=max_is_better,
            props="color:black; font-weight:bold; background-color:yellow;",
        )
        .highlight_min(
            subset=min_is_better,
            props="color:black; font-weight:bold; background-color:yellow;",
        )
        .set_caption("Top {} trials according to {}".format(k, analysis.default_metric))
        .hide_index()
    )
    if save or save_dir:
        if save_dir:
            xl_file = str(Path(save_dir) / "summary_table.xlsx")
        else:
            xl_file = "summary_table.xlsx"
        styled_summary.to_excel(xl_file, engine="openpyxl")
        print("Saved plot table to {}".format(xl_file))
    return summary, styled_summary


def analyze_ray_experiment(exp_dir, default_metric, default_mode):
    from ray.tune import Analysis

    analysis = Analysis(exp_dir, default_metric=default_metric, default_mode=default_mode)

    topk_summary_plot_v2(analysis, 5, save_dir=exp_dir)

    summ, styled = summarize_top_k(analysis, k=10, save_dir=exp_dir)


def count_skipped_configurations(exp_dir):
    skiplog_file_path = Path(exp_dir) / "skipped_configurations.txt"
    if skiplog_file_path.exists():
        with open(skiplog_file_path, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if line == "#" * 80 + "\n":
                    count += 1
        if count % 2 != 0:
            print("WARNING: counts is not divisible by two")
        return count // 2
    else:
        print("Could not find {}".format(str(skiplog_file_path)))
