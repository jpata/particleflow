import os
import stat
import argparse
import glob
import yaml
from mlpf.utils import load_spec, resolve_path

# Configuration
LOCAL_JOBS_DIR = "snakemake_validation"
SPEC_FILE = "particleflow_spec.yaml"
VALIDATION_SPEC_FILE = "validation_key4hep.yaml"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_bash_script(path, content, project_root=None, tmpdir=None):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("export GOTO_NUM_THREADS=1\n")
        f.write("export MKL_NUM_THREADS=1\n")
        f.write("export NUMEXPR_NUM_THREADS=1\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("export OPENBLAS_NUM_THREADS=1\n")
        f.write("export VECLIB_MAXIMUM_THREADS=1\n")
        if tmpdir:
            f.write(f"export TMPDIR={tmpdir}\n")
            f.write(f"export TEMPDIR={tmpdir}\n")
            f.write(f"export TEMP={tmpdir}\n")
            f.write(f"export TMP={tmpdir}\n")
            f.write("mkdir -p $TMPDIR\n")
        if project_root:
            f.write(f"cd {project_root}\n")
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def parse_runtime(runtime_str):
    if isinstance(runtime_str, int):
        return runtime_str
    if runtime_str.endswith("h"):
        return int(runtime_str[:-1]) * 60
    if runtime_str.endswith("m"):
        return int(runtime_str[:-1])
    if runtime_str.endswith("d"):
        return int(runtime_str[:-1]) * 60 * 24
    try:
        return int(runtime_str)
    except ValueError:
        return runtime_str


def get_resource_str(executor, mem, partition, runtime, threads=1, gpus=0, gpu_type=None, slurm_account=None):
    res = {}
    runtime_m = parse_runtime(runtime)
    if executor == "slurm":
        res["mem_mb"] = mem
        res["slurm_partition"] = f'"{partition}"'
        res["runtime"] = runtime_m
        if slurm_account:
            res["slurm_account"] = f'"{slurm_account}"'
        if gpus > 0:
            if gpu_type:
                res["gres"] = f'"gpu:{gpu_type}:{gpus}"'
            else:
                res["gpu"] = gpus
        res["cpus_per_task"] = threads
    else:
        res["mem_mb"] = mem
    return ", ".join([f"{k}={v}" for k, v in res.items()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="cld", help="Scenario name from config")
    args = parser.parse_args()

    # Load main spec and validation config
    spec = load_spec(SPEC_FILE)
    with open(VALIDATION_SPEC_FILE, "r") as f:
        vspec = yaml.safe_load(f)

    if args.scenario not in vspec.get("scenarios", {}):
        raise ValueError(f"Scenario {args.scenario} not found in {VALIDATION_SPEC_FILE}")

    scen = vspec["scenarios"][args.scenario]

    checkpoint = scen.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"No checkpoint found in scenario {args.scenario} in {VALIDATION_SPEC_FILE}")

    checkpoint_abs = os.path.abspath(checkpoint)
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_abs))
    project_root = resolve_path("${project.paths.project_root}", spec)

    # Resolve values from scenario config
    workspace_dir = resolve_path(scen["workspace_dir"], spec)
    container_img = resolve_path(scen["container"], spec)
    detector = scen["detector"]

    # Resources
    res = scen.get("resources", {})
    mem_mb = res.get("mem_mb", 4000)
    runtime = resolve_path(res.get("runtime", "2h"), spec)
    partition = resolve_path(res.get("slurm_partition", "main"), spec)
    executor = spec["project"].get("executor", "slurm")
    slurm_account = spec["project"].get("slurm_account")

    # Experiment name and num_files for directory organization
    exp_name = scen.get("exp_name") or os.path.basename(exp_dir)
    num_files = scen.get("num_files", vspec.get("num_files", -1))

    # Configuration path for evaluator
    config_path = os.path.join(exp_dir, "model_kwargs.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(exp_dir, "hyperparameters.json")

    # Output directory for validation results
    val_out_dir = os.path.join(project_root, "experiments", exp_name, f"validation_{args.scenario}")
    ensure_dir(val_out_dir)

    # Job scripts directory
    jobs_dir = f"{LOCAL_JOBS_DIR}/{exp_name}_{args.scenario}"
    ensure_dir(f"{jobs_dir}/scripts")
    ensure_dir(f"{jobs_dir}/done")

    # 1. Evaluator Script
    eval_script_path = f"{jobs_dir}/scripts/evaluate.sh"
    eval_cmd = (
        f"python3 mlpf/standalone_eval/key4hep/evaluator.py --input $1 --checkpoint {checkpoint_abs} "
        f"--config {config_path} --detector {detector} --outpath $2 --num-events -1"
    )
    write_bash_script(eval_script_path, f"export PYTHONPATH=$(pwd):$PYTHONPATH\n{eval_cmd}", project_root=project_root)

    # 2. Plotting Script
    plot_script_path = f"{jobs_dir}/scripts/plot.sh"
    plot_cmd = "python3 mlpf/standalone_eval/key4hep/plots.py --input $1 --outdir $2"
    write_bash_script(plot_script_path, f"export PYTHONPATH=$(pwd):$PYTHONPATH\n{plot_cmd}", project_root=project_root)

    # Generate Snakefile
    snakefile_path = f"{jobs_dir}/Snakefile"
    targets = []
    rules = ""

    for sample_name, sample_data in scen.get("mc_samples", {}).items():
        process_name = sample_data["process_name"]
        input_dir = os.path.join(workspace_dir, "gen", process_name, "root")

        print(f"Searching for ROOT files in {input_dir}")
        root_files = sorted(glob.glob(os.path.join(input_dir, "*.root")))
        if not root_files:
            root_files = sorted(glob.glob(os.path.join(input_dir, "**/*.root"), recursive=True))

        if not root_files:
            print(f"Warning: No ROOT files found for sample {sample_name} in {input_dir}")
            continue

        if num_files > 0:
            root_files = root_files[:num_files]

        print(f"Found {len(root_files)} ROOT files for {sample_name}.")

        sample_out_dir = os.path.join(val_out_dir, sample_name)
        ensure_dir(sample_out_dir)

        for i, root_file in enumerate(root_files):
            base = os.path.basename(root_file).replace(".root", "")
            parquet_file = os.path.join(sample_out_dir, f"{base}.parquet")
            plot_dir = os.path.join(sample_out_dir, f"plots_{base}")
            done_file = f"{jobs_dir}/done/{sample_name}_{base}.done"

            targets.append(f'"{done_file}"')

            rules += f"""
rule eval_{sample_name}_{i}:
    input:
        root = "{root_file}"
    output:
        parquet = "{parquet_file}"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{eval_script_path} {{input.root}} {{output.parquet}}"

rule plot_{sample_name}_{i}:
    input:
        parquet = "{parquet_file}"
    output:
        done = "{done_file}"
    resources:
        {get_resource_str(executor, 2000, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{plot_script_path} {{input.parquet}} {plot_dir} && touch {{output.done}}"
"""

    with open(snakefile_path, "w") as f:
        f.write("import os\n\n")
        f.write("rule all:\n    input:\n        " + ",\n        ".join(targets) + "\n")
        f.write(rules)

    print(f"Generated validation Snakemake workflow in {jobs_dir}")
    print(f"To run: snakemake -s {snakefile_path} --jobs 4")

    # Write jobs_dir to a file for pixi consumption
    with open(".last_jobs_dir", "w") as f:
        f.write(jobs_dir)


if __name__ == "__main__":
    main()
