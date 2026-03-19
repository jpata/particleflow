import json
import re
import os
import glob


def parse_log(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Extract DSL
    dsl_match = re.search(r"Using DSL: (.*)", content)
    if not dsl_match:
        return None, None

    dsl = dsl_match.group(1).strip()

    # Extract metrics
    metrics = {}
    
    # Looking for final results section
    # Example: val_jet_iqr     : 1.709958 ± 0.000408 (var)
    metric_patterns = {
        "val_jet_iqr": r"val_jet_iqr\s+:\s+([\d\.]+)",
        "val_jet_matched_frac": r"val_jet_matched_frac:\s+([\d\.]+)",
        "runtime_cpu_ms": r"runtime_cpu_ms\s+:\s+([\d\.]+)",
        "runtime_gpu_ms": r"runtime_gpu_ms\s+:\s+([\d\.]+)",
        "peak_vram_mb": r"peak_vram_mb\s+:\s+([\d\.]+)",
        "num_params_M": r"num_params_M:\s+([\d\.]+)",
    }

    for name, pattern in metric_patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[name] = float(match.group(1))

    return dsl, metrics


def main():
    log_files = glob.glob("logs/good/log*.txt")
    population_metrics = {}

    print(f"Found {len(log_files)} log files.")

    for log_file in log_files:
        try:
            dsl, metrics = parse_log(log_file)
            if dsl and metrics:
                print(f"Parsed {log_file}: {dsl}")
                population_metrics[dsl] = metrics
            else:
                print(f"Could not parse {log_file}")
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

    with open("population_metrics.json", "w") as f:
        json.dump(population_metrics, f, indent=4)

    print(f"\nSaved metrics for {len(population_metrics)} configurations to population_metrics.json")


if __name__ == "__main__":
    main()
