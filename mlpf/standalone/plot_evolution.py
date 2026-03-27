import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mlpf.standalone.run_evolution import calculate_fitness

# Set modern visual style: white background, no grid
sns.set_theme(style="white", context="talk", font="sans-serif")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Clear Sans"],
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
    }
)


def load_metrics(file_path, gen_num):
    with open(file_path, "r") as f:
        data = json.load(f)

    models = []
    for k, v in data.items():
        if "val_jet_iqr" in v and v["val_jet_iqr"] > 0:
            val_loss = v.get("val_loss", 10.0)
            matched_frac = v.get("val_jet_matched_frac", 0)
            runtime_cpu = v.get("runtime_cpu_ms", 1000.0)
            peak_vram = v.get("peak_vram_mb", 0)

            # Total fitness
            fitness, _ = calculate_fitness(v)

            models.append(
                {
                    "key": k,
                    "fitness": fitness,
                    "val_loss": val_loss,
                    "match_frac": matched_frac,
                    "gpu_runtime": v.get("runtime_gpu_ms", 0),
                    "cpu_runtime": runtime_cpu,
                    "peak_vram": peak_vram,
                    "gen": gen_num,
                }
            )
    return models


def main():
    file_pattern = "logs/evolution/gen_*_metrics.json"
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found matching {file_pattern}")
        return

    # Sort files by generation number
    files.sort(key=lambda x: int(re.search(r"gen_(\d+)_metrics.json", x).group(1)))

    gen_numbers = [int(re.search(r"gen_(\d+)_metrics.json", f).group(1)) for f in files]
    generations = [f"Gen {n}" for n in gen_numbers]

    all_models_by_gen = []
    averages = []
    top_ks = []
    k = 5

    for i, f in enumerate(files):
        gen_models = load_metrics(f, gen_numbers[i])
        all_models_by_gen.append(gen_models)

        fitness = [m["fitness"] for m in gen_models]
        if fitness:
            averages.append(np.mean(fitness))
            top_ks.append(np.mean(sorted(fitness, reverse=True)[:k]))
        else:
            averages.append(0)
            top_ks.append(0)

    # Plotting setup: 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(22, 26))
    axes_flat = axes.flatten()

    # Font sizes
    title_fs = 34
    label_fs = 30
    tick_fs = 24
    legend_fs = 22

    # Fixed colors for Fitness Evolution
    avg_color = "#1f77b4"  # Steel Blue
    topk_color = "#d62728"  # Crimson

    # Calculate global limits for stable axes in animation
    all_fitness = [m["fitness"] for gen in all_models_by_gen for m in gen]
    all_match_frac = [m["match_frac"] for gen in all_models_by_gen for m in gen]
    all_gpu_runtime = [m["gpu_runtime"] for gen in all_models_by_gen for m in gen]
    all_cpu_runtime = [m["cpu_runtime"] for gen in all_models_by_gen for m in gen]
    all_val_loss = [m["val_loss"] for gen in all_models_by_gen for m in gen]
    all_peak_vram = [m["peak_vram"] for gen in all_models_by_gen for m in gen]

    fit_lim = (0, max(all_fitness + top_ks) * 1.1)
    mf_lim = (max(0, min(all_match_frac) * 0.9), min(1.0, max(all_match_frac) * 1.1))
    gr_lim = (0, max(all_gpu_runtime) * 1.1)
    cr_lim = (0, max(all_cpu_runtime) * 1.1)
    vl_lim = (0, max(all_val_loss) * 1.1)
    pv_lim = (0, max(all_peak_vram) * 1.1)

    colors = sns.color_palette("viridis", n_colors=len(generations))

    def update(frame):
        for ax in axes_flat:
            ax.clear()
            ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        fig.legends.clear()

        ax1, ax2, ax3, ax4, ax5, ax6 = axes_flat

        # Plot 1: Fitness Evolution
        curr_x = np.arange(frame + 1)
        ax1.plot(curr_x, averages[: frame + 1], marker="o", linestyle="-", linewidth=5, markersize=14, label="Average Fitness", color=avg_color)
        ax1.plot(curr_x, top_ks[: frame + 1], marker="s", linestyle="--", linewidth=5, markersize=14, label=f"Top-{k} Fitness", color=topk_color)
        ax1.set_ylabel("Fitness (Full Metric)", fontsize=label_fs, fontweight="bold")
        ax1.set_title(f"Fitness Evolution (up to Gen {gen_numbers[frame]})", fontsize=title_fs, fontweight="bold", pad=20)
        ax1.set_xticks(np.arange(len(generations)))
        ax1.set_xticklabels(generations, rotation=45, fontsize=tick_fs)
        ax1.set_xlim(-0.5, len(generations) - 0.5)
        ax1.set_ylim(fit_lim)
        ax1.legend(loc="upper left", fontsize=legend_fs, frameon=True, shadow=True)
        for i in range(frame + 1):
            ax1.text(curr_x[i], averages[i], f"{averages[i]:.3f}", ha="center", va="bottom", fontsize=tick_fs, fontweight="bold", color=avg_color)
            ax1.text(curr_x[i], top_ks[i], f"{top_ks[i]:.3f}", ha="center", va="bottom", fontsize=tick_fs, fontweight="bold", color=topk_color)

        # Scatter Plots
        for i in range(frame + 1):
            gen_models = all_models_by_gen[i]
            if gen_models:
                fitness = [m["fitness"] for m in gen_models]
                match_frac = [m["match_frac"] for m in gen_models]
                gpu_runtime = [m["gpu_runtime"] for m in gen_models]
                cpu_runtime = [m["cpu_runtime"] for m in gen_models]
                val_loss = [m["val_loss"] for m in gen_models]
                peak_vram = [m["peak_vram"] for m in gen_models]

                ax2.scatter(match_frac, fitness, color=colors[i], label=generations[i], alpha=0.8, s=200, edgecolor="white", linewidth=1.5)
                ax3.scatter(gpu_runtime, fitness, color=colors[i], label=generations[i], alpha=0.8, s=200, edgecolor="white", linewidth=1.5)
                ax4.scatter(cpu_runtime, fitness, color=colors[i], label=generations[i], alpha=0.8, s=200, edgecolor="white", linewidth=1.5)
                ax5.scatter(val_loss, fitness, color=colors[i], label=generations[i], alpha=0.8, s=200, edgecolor="white", linewidth=1.5)
                ax6.scatter(peak_vram, fitness, color=colors[i], label=generations[i], alpha=0.8, s=200, edgecolor="white", linewidth=1.5)

        ax2.set_xlabel("Match Fraction (val_jet_matched_frac)", fontsize=label_fs, fontweight="bold")
        ax2.set_ylabel("Fitness", fontsize=label_fs, fontweight="bold")
        ax2.set_title("Fitness vs Match Fraction", fontsize=title_fs, fontweight="bold", pad=15)
        ax2.set_xlim(mf_lim)
        ax2.set_ylim(fit_lim)

        ax3.set_xlabel("GPU Runtime (ms)", fontsize=label_fs, fontweight="bold")
        ax3.set_ylabel("Fitness", fontsize=label_fs, fontweight="bold")
        ax3.set_title("Fitness vs GPU Runtime", fontsize=title_fs, fontweight="bold", pad=15)
        ax3.set_xlim(gr_lim)
        ax3.set_ylim(fit_lim)

        ax4.set_xlabel("CPU Runtime (ms)", fontsize=label_fs, fontweight="bold")
        ax4.set_ylabel("Fitness", fontsize=label_fs, fontweight="bold")
        ax4.set_title("Fitness vs CPU Runtime", fontsize=title_fs, fontweight="bold", pad=15)
        ax4.set_xlim(cr_lim)
        ax4.set_ylim(fit_lim)

        ax5.set_xlabel("Validation Loss", fontsize=label_fs, fontweight="bold")
        ax5.set_ylabel("Fitness", fontsize=label_fs, fontweight="bold")
        ax5.set_title("Fitness vs Val Loss", fontsize=title_fs, fontweight="bold", pad=15)
        ax5.set_xlim(vl_lim)
        ax5.set_ylim(fit_lim)

        ax6.set_xlabel("Peak VRAM (MB)", fontsize=label_fs, fontweight="bold")
        ax6.set_ylabel("Fitness", fontsize=label_fs, fontweight="bold")
        ax6.set_title("Fitness vs Peak VRAM", fontsize=title_fs, fontweight="bold", pad=15)
        ax6.set_xlim(pv_lim)
        ax6.set_ylim(fit_lim)

        # Remove grid and despine
        for ax in axes_flat:
            ax.grid(False)
            sns.despine(ax=ax)

        # Common legend for scatter plots
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.99, 0.5), fontsize=legend_fs, frameon=True, shadow=True)

        plt.tight_layout(rect=[0, 0, 0.92, 1])

    # Create animation
    print("Creating animation...")
    ani = FuncAnimation(fig, update, frames=len(generations), repeat=False)

    # Save the final state as PNG
    update(len(generations) - 1)
    plt.savefig("fitness_evolution.png")
    print("Static plot saved to fitness_evolution.png")

    # Save animation as GIF
    try:
        ani.save("fitness_evolution.gif", writer="pillow", fps=1)
        print("Animation saved to fitness_evolution.gif")
    except Exception as e:
        print(f"Could not save GIF: {e}")

    # Print top 10 models
    print("\nTop 10 models by fitness:")
    all_models = [m for gen in all_models_by_gen for m in gen]
    all_models.sort(key=lambda x: x["fitness"], reverse=True)
    for i, m in enumerate(all_models[:10]):
        print(
            f"{i+1:2d}. Gen {m['gen']:2d} - Fitness: {m['fitness']:.4f} - Val Loss: {m['val_loss']:.4f} - Match Frac: {m['match_frac']:.4f} - VRAM: {m['peak_vram']:.1f} MB - Key: {m['key']}"
        )


if __name__ == "__main__":
    main()
