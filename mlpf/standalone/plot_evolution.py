import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re


def load_metrics(file_path, gen_num):
    with open(file_path, "r") as f:
        data = json.load(f)

    models = []
    for k, v in data.items():
        if "val_jet_iqr" in v and v["val_jet_iqr"] > 0:
            iqr = v["val_jet_iqr"]
            matched_frac = v.get("val_jet_matched_frac", 0)
            runtime_cpu = v.get("runtime_cpu_ms", 1000.0)
            val_loss = v.get("val_loss", 10.0)
            peak_vram = v.get("peak_vram_mb", 0)

            # Individual terms for fitness
            term_matching = matched_frac
            term_iqr = 1.0 / max(iqr, 0.01)
            term_loss = 1.0 / (1.0 + val_loss)
            term_runtime = 1.0 / (1.0 + runtime_cpu / 1000.0)

            # Total fitness
            fitness = term_matching * term_iqr * term_loss * term_runtime

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
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    # Plot 1: Fitness Evolution
    x = np.arange(len(generations))
    ax1.plot(x, averages, marker="o", linestyle="-", label="Average Fitness")
    ax1.plot(x, top_ks, marker="s", linestyle="--", label=f"Top-{k} Fitness")
    ax1.set_ylabel("Fitness (Full Metric)")
    ax1.set_title("Fitness Evolution across Generations")
    ax1.set_xticks(x)
    ax1.set_xticklabels(generations, rotation=45)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    for i, (avg, top) in enumerate(zip(averages, top_ks)):
        ax1.text(x[i], avg, f"{avg:.3f}", ha="center", va="bottom")
        ax1.text(x[i], top, f"{top:.3f}", ha="center", va="bottom")

    # Scatter Plots: Colors by Generation
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
    for i, gen_name in enumerate(generations):
        gen_models = all_models_by_gen[i]
        if gen_models:
            fitness = [m["fitness"] for m in gen_models]
            match_frac = [m["match_frac"] for m in gen_models]
            gpu_runtime = [m["gpu_runtime"] for m in gen_models]
            cpu_runtime = [m["cpu_runtime"] for m in gen_models]
            val_loss = [m["val_loss"] for m in gen_models]
            peak_vram = [m["peak_vram"] for m in gen_models]

            ax2.scatter(match_frac, fitness, color=colors[i], label=gen_name, alpha=0.6)
            ax3.scatter(gpu_runtime, fitness, color=colors[i], label=gen_name, alpha=0.6)
            ax4.scatter(cpu_runtime, fitness, color=colors[i], label=gen_name, alpha=0.6)
            ax5.scatter(val_loss, fitness, color=colors[i], label=gen_name, alpha=0.6)
            ax6.scatter(peak_vram, fitness, color=colors[i], label=gen_name, alpha=0.6)

    ax2.set_xlabel("Match Fraction (val_jet_matched_frac)")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness vs Match Fraction")

    ax3.set_xlabel("GPU Runtime (ms)")
    ax3.set_ylabel("Fitness")
    ax3.set_title("Fitness vs GPU Runtime")

    ax4.set_xlabel("CPU Runtime (ms)")
    ax4.set_ylabel("Fitness")
    ax4.set_title("Fitness vs CPU Runtime")

    ax5.set_xlabel("Validation Loss")
    ax5.set_ylabel("Fitness")
    ax5.set_title("Fitness vs Val Loss")

    ax6.set_xlabel("Peak VRAM (MB)")
    ax6.set_ylabel("Fitness")
    ax6.set_title("Fitness vs Peak VRAM")

    # Common legend for scatter plots
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for legend
    plt.savefig("fitness_evolution.png")
    print("Plot saved to fitness_evolution.png")

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
