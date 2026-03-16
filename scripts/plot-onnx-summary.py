import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import seaborn as sns

matplotlib.use("Agg")
mplhep.style.use("CMS")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot runtime scaling from ONNX summary JSONs.")
    parser.add_argument("--indir", type=str, required=True, help="Directory containing config subfolders with summary.json")
    parser.add_argument("--outdir", type=str, default="./onnx_plots", help="Output directory for plots")
    return parser.parse_args()


def load_data(indir):
    all_data = {}
    model_metadata = {}
    # Iterate over subdirectories (e.g., standard, linear, gnn-lsh)
    for subdir in sorted(os.listdir(indir)):
        summary_path = os.path.join(indir, subdir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
                system_info = data.get("system", {})

                # Store model-level metadata like train_loss
                model_metadata[subdir] = {"train_loss": data.get("train_loss"), "valid_loss": data.get("valid_loss")}

                for scenario, runs in data.get("scenarios", {}).items():
                    # Combine model and scenario for a unique label
                    full_label = f"{subdir}: {scenario}"
                    if full_label not in all_data:
                        all_data[full_label] = {"sizes": [], "runtimes": [], "ooms": [], "maes": [], "model": subdir, "scenario": scenario}

                    for run in runs:
                        if run["runtime"] is not None:
                            all_data[full_label]["sizes"].append(run["size"])
                            all_data[full_label]["runtimes"].append(run["runtime"] * 1000.0)  # ms
                        if run.get("mae") is not None:
                            all_data[full_label]["maes"].append(run["mae"])
                        if run["oom"]:
                            all_data[full_label]["ooms"].append(run["size"])

    return all_data, model_metadata, system_info


def plot_mae_vs_runtime(data, outdir, system_info):
    plot_data = []

    # Identify unique models
    models = sorted(list(set(v["model"] for v in data.values())))

    for model in models:
        # Find all scenarios for this model
        model_scenarios = {k: v for k, v in data.items() if v["model"] == model}

        if not model_scenarios:
            continue

        # Find the fastest scenario (lowest mean runtime)
        best_scenario_key = None
        min_mean_rt = float("inf")

        for key, vals in model_scenarios.items():
            if len(vals["runtimes"]) > 0:
                mean_rt = np.mean(vals["runtimes"])
                if mean_rt < min_mean_rt:
                    min_mean_rt = mean_rt
                    best_scenario_key = key

        if best_scenario_key:
            vals = data[best_scenario_key]
            if len(vals["maes"]) > 0:
                plot_data.append(
                    {
                        "Model": model,
                        "Fastest Scenario": vals["scenario"],
                        "Avg Runtime [ms]": np.mean(vals["runtimes"]),
                        "Std Runtime [ms]": np.std(vals["runtimes"]),
                        "Avg MAE": np.mean(vals["maes"]),
                        "Std MAE": np.std(vals["maes"]),
                    }
                )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Use different markers for models
    unique_models = df["Model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_to_color = {model: colors[i] for i, model in enumerate(unique_models)}

    for i, row in df.iterrows():
        plt.errorbar(
            row["Avg Runtime [ms]"],
            row["Avg MAE"],
            xerr=row["Std Runtime [ms]"],
            yerr=row["Std MAE"],
            fmt="o",
            markersize=10,
            capsize=5,
            color=model_to_color[row["Model"]],
            label=row["Model"],
        )

        # Annotate points
        plt.text(
            row["Avg Runtime [ms]"],
            row["Avg MAE"],
            f" {row['Model']}\n ({row['Fastest Scenario']})",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=9,
        )

    # Handle legend to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Avg Runtime [ms]")
    plt.ylabel("MAE")
    plt.title("Numerical Error (MAE) vs. Inference Runtime", y=1.05)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Log scales can be useful for error
    plt.yscale("log")

    # Add margins to ensure space for labels and error bars
    plt.margins(x=0.3, y=0.2)

    # Add system info text
    system_text = f"Device: {system_info.get('device', 'Unknown')}\nCPU: {system_info.get('cpu', 'Unknown')}\nGPU: {system_info.get('gpu', 'Unknown')}\nPyTorch: {system_info.get('pytorch_version', 'Unknown')}\nONNX Runtime: {system_info.get('onnxruntime_version', 'Unknown')}"
    plt.text(
        0.02,
        0.98,
        system_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mae_vs_runtime.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "mae_vs_runtime.png"), bbox_inches="tight")


def plot_loss_vs_runtime(data, model_metadata, outdir, system_info):
    plot_data = []

    # Identify unique models
    models = sorted(list(set(v["model"] for v in data.values())))

    for model in models:
        # Find all scenarios for this model
        model_scenarios = {k: v for k, v in data.items() if v["model"] == model}

        if not model_scenarios:
            continue

        # Find the fastest scenario (lowest mean runtime)
        best_scenario_key = None
        min_mean_rt = float("inf")
        std_rt = 0

        for key, vals in model_scenarios.items():
            if len(vals["runtimes"]) > 0:
                mean_rt = np.mean(vals["runtimes"])
                if mean_rt < min_mean_rt:
                    min_mean_rt = mean_rt
                    std_rt = np.std(vals["runtimes"])
                    best_scenario_key = key

        if best_scenario_key:
            train_loss = model_metadata[model].get("train_loss")
            # train_loss might be a list (history), take the last value if it's a list
            if isinstance(train_loss, list) and len(train_loss) > 0:
                train_loss = train_loss[-1]

            if train_loss is not None:
                plot_data.append(
                    {
                        "Model": model,
                        "Fastest Scenario": data[best_scenario_key]["scenario"],
                        "Avg Runtime [ms]": min_mean_rt,
                        "Std Runtime [ms]": std_rt,
                        "Train Loss": train_loss,
                    }
                )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Use different markers for models
    unique_models = df["Model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_to_color = {model: colors[i] for i, model in enumerate(unique_models)}

    for i, row in df.iterrows():
        plt.errorbar(
            row["Avg Runtime [ms]"],
            row["Train Loss"],
            xerr=row["Std Runtime [ms]"],
            fmt="o",
            markersize=10,
            capsize=5,
            color=model_to_color[row["Model"]],
            label=row["Model"],
        )

        # Annotate points
        plt.text(
            row["Avg Runtime [ms]"],
            row["Train Loss"],
            f" {row['Model']}\n ({row['Fastest Scenario']})",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=9,
        )

    # Handle legend to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Set plot limits with generous margin for labels
    max_x = (df["Avg Runtime [ms]"] + df["Std Runtime [ms]"]).max()
    max_y = df["Train Loss"].max()
    plt.xlim(0, 1.8 * max_x)
    plt.ylim(0, 1.8 * max_y)

    plt.xlabel("Avg Runtime [ms]")
    plt.ylabel("Train Loss")
    plt.title("Train Loss vs. Inference Runtime", y=1.05)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add system info text
    system_text = f"Device: {system_info.get('device', 'Unknown')}\nCPU: {system_info.get('cpu', 'Unknown')}\nGPU: {system_info.get('gpu', 'Unknown')}\nPyTorch: {system_info.get('pytorch_version', 'Unknown')}\nONNX Runtime: {system_info.get('onnxruntime_version', 'Unknown')}"
    plt.text(
        0.02,
        0.98,
        system_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_vs_runtime.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "loss_vs_runtime.png"), bbox_inches="tight")


def plot_violin_summary(data, outdir, system_info):
    baseline = "PT_MATH_FP32"
    plot_data = []

    # Identify unique models
    models = sorted(list(set(v["model"] for v in data.values())))

    for model in models:
        # Find all scenarios for this model
        model_scenarios = {k: v for k, v in data.items() if v["model"] == model}

        if not model_scenarios:
            continue

        # 1. Collect baseline (PT_MATH_FP32)
        baseline_key = f"{model}: {baseline}"
        if baseline_key in model_scenarios:
            vals = model_scenarios[baseline_key]
            for rt in vals["runtimes"]:
                plot_data.append({"Model": model, "Scenario": "Baseline (PT_MATH_FP32)", "Inference Time [ms]": rt})

        # 2. Find the fastest scenario (lowest mean runtime)
        best_scenario = None
        min_mean_rt = float("inf")

        for key, vals in model_scenarios.items():
            if len(vals["runtimes"]) > 0:
                mean_rt = np.mean(vals["runtimes"])
                if mean_rt < min_mean_rt:
                    min_mean_rt = mean_rt
                    best_scenario = key

        if best_scenario:
            # Add fastest scenario to plot_data, but only if it's not the baseline itself
            # or if we want to show it explicitly.
            scenario_name = data[best_scenario]["scenario"]
            label = f"Fastest ({scenario_name})"
            vals = data[best_scenario]
            for rt in vals["runtimes"]:
                plot_data.append({"Model": model, "Scenario": label, "Inference Time [ms]": rt})

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x="Model", y="Inference Time [ms]", hue="Scenario", split=False, inner="quart")

    plt.xlabel("Model")
    plt.ylabel("Inference Time [ms]")
    plt.title("Runtime Comparison: Baseline vs. Fastest Scenario", y=1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add system info text
    ax = plt.gca()
    system_text = f"Device: {system_info.get('device', 'Unknown')}\nCPU: {system_info.get('cpu', 'Unknown')}\nGPU: {system_info.get('gpu', 'Unknown')}\nPyTorch: {system_info.get('pytorch_version', 'Unknown')}\nONNX Runtime: {system_info.get('onnxruntime_version', 'Unknown')}"
    plt.text(
        0.02,
        0.98,
        system_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_violin_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "runtime_violin_comparison.png"), bbox_inches="tight")

    # Also log scale version
    plt.yscale("log")
    plt.title("Runtime Comparison: Baseline vs. Fastest Scenario (Log Scale)", y=1.05)
    plt.savefig(os.path.join(outdir, "runtime_violin_comparison_log.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "runtime_violin_comparison_log.png"), bbox_inches="tight")


def plot_bar_summary(data, outdir, system_info):
    baseline_label = "Baseline (PT_MATH_FP32)"
    fastest_label = "Fastest Scenario"
    plot_data = []

    models = sorted(list(set(v["model"] for v in data.values())))

    for model in models:
        model_scenarios = {k: v for k, v in data.items() if v["model"] == model}
        if not model_scenarios:
            continue

        # Find Fastest Scenario name for the label
        best_scenario_key = None
        min_mean_rt = float("inf")
        for key, vals in model_scenarios.items():
            if len(vals["runtimes"]) > 0:
                mean_rt = np.mean(vals["runtimes"])
                if mean_rt < min_mean_rt:
                    min_mean_rt = mean_rt
                    best_scenario_key = key

        if best_scenario_key:
            scenario_name = data[best_scenario_key]["scenario"]
            full_model_label = f"{model}\n(Fastest: {scenario_name})"
        else:
            full_model_label = model

        # Baseline data
        baseline_key = f"{model}: PT_MATH_FP32"
        if baseline_key in model_scenarios:
            vals = model_scenarios[baseline_key]
            for rt in vals["runtimes"]:
                plot_data.append({"Model": full_model_label, "Scenario": baseline_label, "Inference Time [ms]": rt})

        # Fastest data
        if best_scenario_key:
            vals = data[best_scenario_key]
            for rt in vals["runtimes"]:
                plot_data.append({"Model": full_model_label, "Scenario": fastest_label, "Inference Time [ms]": rt})

    if not plot_data:
        return
    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(16, 10))
    # Horizontal bar plot: swap x and y
    ax = sns.barplot(
        data=df, y="Model", x="Inference Time [ms]", hue="Scenario", hue_order=[baseline_label, fastest_label], capsize=0.1, errorbar="sd", gap=0
    )

    plt.xlabel("Inference Time [ms]")
    plt.ylabel("Model")
    plt.title("Mean Runtime Comparison: Baseline vs. Fastest Scenario", y=1.05)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    system_text = f"Device: {system_info.get('device', 'Unknown')}\nCPU: {system_info.get('cpu', 'Unknown')}\nGPU: {system_info.get('gpu', 'Unknown')}\nPyTorch: {system_info.get('pytorch_version', 'Unknown')}\nONNX Runtime: {system_info.get('onnxruntime_version', 'Unknown')}"
    plt.text(
        0.02,
        0.98,
        system_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_bar_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "runtime_bar_comparison.png"), bbox_inches="tight")

    plt.xscale("log")
    plt.title("Mean Runtime Comparison: Baseline vs. Fastest Scenario (Log Scale)", y=1.05)
    plt.savefig(os.path.join(outdir, "runtime_bar_comparison_log.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "runtime_bar_comparison_log.png"), bbox_inches="tight")


def save_text_summary(data, model_metadata, outdir):
    summary_lines = []
    summary_lines.append("=== ONNX Runtime Summary ===")

    models = sorted(list(set(v["model"] for v in data.values())))

    for model in models:
        summary_lines.append(f"\nModel: {model}")

        # Train Loss
        train_loss = model_metadata[model].get("train_loss")
        if isinstance(train_loss, list) and len(train_loss) > 0:
            train_loss = train_loss[-1]
        if train_loss is not None:
            summary_lines.append(f"  Train Loss: {train_loss:.4g}")
        else:
            summary_lines.append("  Train Loss: None")

        # Best Scenario
        model_scenarios = {k: v for k, v in data.items() if v["model"] == model}
        best_scenario_key = None
        min_mean_rt = float("inf")

        for key, vals in model_scenarios.items():
            if len(vals["runtimes"]) > 0:
                mean_rt = np.mean(vals["runtimes"])
                if mean_rt < min_mean_rt:
                    min_mean_rt = mean_rt
                    best_scenario_key = key

        if best_scenario_key:
            scenario_name = data[best_scenario_key]["scenario"]
            summary_lines.append(f"  Best Scenario: {scenario_name}")
            summary_lines.append(f"  Best Avg Runtime: {min_mean_rt:.2f} ms")

            # Add Avg MAE for the best scenario
            maes = data[best_scenario_key].get("maes", [])
            if maes:
                avg_mae = np.mean(maes)
                summary_lines.append(f"  Best Scenario Avg MAE: {avg_mae:.4g}")

        # OOMs
        total_ooms = 0
        min_oom_size = float("inf")
        for key, vals in model_scenarios.items():
            total_ooms += len(vals["ooms"])
            if vals["ooms"]:
                min_oom_size = min(min_oom_size, min(vals["ooms"]))

        if total_ooms > 0:
            summary_lines.append(f"  Number of OOMs: {total_ooms}")
            summary_lines.append(f"  Minimum event size for OOM: {min_oom_size}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary_text)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data, model_metadata, system_info = load_data(args.indir)

    if not data:
        print(f"No summary.json files found in {args.indir}")
        return

    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Use unique markers for each model
    unique_models = sorted(list(set(v["model"] for v in data.values())))
    markers = ["o", "s", "D", "v", "^", "<", ">", "p", "*", "h"]
    model_to_marker = {model: markers[i % len(markers)] for i, model in enumerate(unique_models)}

    # Use consistent colors for scenarios if possible, otherwise use a map
    unique_scenarios = sorted(list(set(v["scenario"] for v in data.values())))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_scenarios)))
    scenario_to_color = {sc: colors[i % len(colors)] for i, sc in enumerate(unique_scenarios)}

    for full_label, vals in sorted(data.items()):
        sizes = np.array(vals["sizes"])
        runtimes = np.array(vals["runtimes"])
        color = scenario_to_color[vals["scenario"]]
        marker = model_to_marker[vals["model"]]

        if len(sizes) > 0:
            # Sort by size for cleaner trend lines
            idx = np.argsort(sizes)
            plt.scatter(sizes[idx], runtimes[idx], label=full_label, color=color, marker=marker, alpha=0.6, s=30)

            # Add a simple linear fit/trend line
            if len(sizes) > 1:
                z = np.polyfit(sizes, runtimes, 1)
                p = np.poly1d(z)
                x_range = np.linspace(sizes.min(), sizes.max(), 100)
                plt.plot(x_range, p(x_range), color=color, linestyle="--", lw=1, alpha=0.4)

        # Plot OOMs if any
        if vals["ooms"]:
            oom_sizes = np.array(vals["ooms"])
            plt.scatter(oom_sizes, [plt.ylim()[1] * 0.95] * len(oom_sizes), marker="x", color=color, label=f"{full_label} (OOM)")

    plt.xlabel("Event size (number of elements)")
    plt.ylabel("Inference time [ms]")
    plt.title("Runtime Scaling vs. Event Size", y=1.05)

    # Add system info text
    system_text = f"Device: {system_info.get('device', 'Unknown')}\nCPU: {system_info.get('cpu', 'Unknown')}\nGPU: {system_info.get('gpu', 'Unknown')}\nPyTorch: {system_info.get('pytorch_version', 'Unknown')}\nONNX Runtime: {system_info.get('onnxruntime_version', 'Unknown')}"
    plt.text(
        0.02,
        0.98,
        system_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.margins(0.1)
    plt.tight_layout()

    plot_path = os.path.join(args.outdir, "runtime_scaling_summary.pdf")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.savefig(plot_path.replace(".pdf", ".png"), bbox_inches="tight")

    # Log-log plot
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Runtime Scaling vs. Event Size (Log-Log)", y=1.05)
    log_plot_path = os.path.join(args.outdir, "runtime_scaling_summary_log.pdf")
    plt.savefig(log_plot_path, bbox_inches="tight")
    plt.savefig(log_plot_path.replace(".pdf", ".png"), bbox_inches="tight")

    # Violin plot summary
    plot_violin_summary(data, args.outdir, system_info)

    # Bar plot summary
    plot_bar_summary(data, args.outdir, system_info)

    # Loss vs. Runtime plot
    plot_loss_vs_runtime(data, model_metadata, args.outdir, system_info)

    # MAE vs. Runtime plot
    plot_mae_vs_runtime(data, args.outdir, system_info)

    # Text summary
    save_text_summary(data, model_metadata, args.outdir)

    print(f"Summary plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
