import os
import stat
import yaml
import argparse
from mlpf.utils import load_spec, resolve_path

# Configuration
LOCAL_JOBS_DIR = "snakemake_validation"
SPEC_FILE = "particleflow_spec.yaml"
VALIDATION_SPEC_FILE = "validation_cms.yaml"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_bash_script(path, content):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        # Ensure all output is unbuffered
        f.write("export PYTHONUNBUFFERED=1\n")
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def get_resource_str(executor, mem, partition, runtime, threads=1, gpus=0, gpu_type=None, mem_per_gpu=0, slurm_account=None):
    res = {}
    if executor == "slurm":
        res["mem_mb"] = mem
        if gpus > 0 and mem_per_gpu > 0:
            res["mem_per_gpu"] = mem_per_gpu
        res["slurm_partition"] = f'"{partition}"'
        res["runtime"] = f'"{runtime}"'
        if slurm_account:
            res["slurm_account"] = f'"{slurm_account}"'
        if gpus > 0:
            if gpu_type:
                res["gres"] = f'"gpu:{gpu_type}:{gpus}"'
            else:
                res["gpu"] = gpus
        res["cpus_per_task"] = threads
        res["threads"] = threads
    elif executor == "condor":

        res["mem_mb"] = mem
        res["job_flavour"] = f'"{partition}"'
        res["runtime"] = f'"{runtime}"'
        if gpus > 0:
            res["request_gpus"] = gpus
    else:
        res["mem_mb"] = mem

    return ", ".join([f"{k}={v}" for k, v in res.items()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=VALIDATION_SPEC_FILE, help="Validation spec file (yaml)")
    parser.add_argument("--scenario", type=str, default="cms_run3", help="Scenario name from config")
    args = parser.parse_args()

    # Load main spec and validation config
    spec = load_spec(SPEC_FILE)
    with open(args.config, "r") as f:
        vspec = yaml.safe_load(f)

    if args.scenario not in vspec.get("scenarios", {}):
        raise ValueError(f"Scenario {args.scenario} not found in {args.config}")

    scen = vspec["scenarios"][args.scenario]

    production = scen["production"]
    if production not in spec["productions"]:
        raise ValueError(f"Production {production} not found in {SPEC_FILE}")

    executor = spec["project"].get("executor", "slurm")
    slurm_account = spec["project"].get("slurm_account")
    main_container_img = spec["project"].get("container")

    # Use container from scenario if provided, else use project container
    container_img = resolve_path(scen.get("container", main_container_img), spec)

    # Resolve paths using both spec and scenario (scenario may contain ${workspace_dir})
    # We first resolve workspace_dir as it's used in others
    workspace_dir = resolve_path(scen["workspace_dir"], spec)
    # Put it back into spec for further resolution
    spec["workspace_dir"] = workspace_dir

    val_dir = resolve_path(scen["val_dir"], spec)
    val_data_dir = resolve_path(scen["val_data_dir"], spec)
    output_dir = resolve_path(scen["output_dir"], spec)
    ensure_dir(output_dir)

    # Resource requirements
    res = scen.get("resources", {})
    mem_mb = res.get("mem_mb", 8000)
    runtime = resolve_path(res.get("runtime", "2h"), spec)
    partition = resolve_path(res.get("slurm_partition", "main"), spec)

    exp_name = f"validation_{args.scenario}"
    jobs_dir = f"{LOCAL_JOBS_DIR}/{exp_name}"
    ensure_dir(jobs_dir)
    ensure_dir(f"{jobs_dir}/prep")
    ensure_dir(f"{jobs_dir}/corr")
    ensure_dir(f"{jobs_dir}/plots")

    final_targets = []
    rules_content = ""

    all_samples = {}
    if "mc_samples" in scen:
        for skey, sdata in scen["mc_samples"].items():
            all_samples[skey] = {
                "pf_dir": f"{val_dir}/pf/{sdata['output_subdir']}/{sdata['process_name']}",
                "mlpf_dir": f"{val_dir}/mlpf/{sdata['output_subdir']}/{sdata['process_name']}",
                "is_data": False,
            }

    if "data_samples" in scen:
        for skey, sdata in scen["data_samples"].items():
            all_samples[skey] = {
                "pf_dir": f"{val_data_dir}/pf/{sdata['output_subdir']}/{sdata['process_name']}",
                "mlpf_dir": f"{val_data_dir}/mlpf/{sdata['output_subdir']}/{sdata['process_name']}",
                "is_data": True,
            }

    # 1. Data Preparation (only for MC samples, data is handled by cmssw_validation_data)
    for sample, paths in all_samples.items():
        if paths["is_data"]:
            continue
        prep_id = f"prep_{sample}"
        prep_script_path = f"{jobs_dir}/prep/{prep_id}.sh"
        prep_sentinel = f"{jobs_dir}/prep/{prep_id}.done"

        cmd = (
            "PYTHONPATH=. python3 mlpf/plotting/data_preparation.py "
            + f"--input-pf {paths['pf_dir']} --input-mlpf {paths['mlpf_dir']} "
            + f"--sample {sample} --output-dir {output_dir}"
        )
        write_bash_script(prep_script_path, cmd)

        rules_content += f"""
rule {prep_id}:
    output:
        "{prep_sentinel}",
        "{output_dir}/{sample}_pf.parquet",
        "{output_dir}/{sample}_mlpf.parquet"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{prep_script_path} && touch {{output[0]}}"
"""

    # 2. Jet Corrections (only for correction_sample)
    corr_sample = scen["correction_sample"]
    for jet_type in scen["jet_types"]:
        corr_id = f"corr_{corr_sample}_{jet_type}"
        corr_script_path = f"{jobs_dir}/corr/{corr_id}.sh"
        corr_sentinel = f"{jobs_dir}/corr/{corr_id}.done"
        jec_file = f"{output_dir}/jec_{jet_type}_{corr_sample}.npz"

        cmd = (
            "PYTHONPATH=. python3 mlpf/plotting/corrections.py "
            + f"--input-pf-parquet {output_dir}/{corr_sample}_pf.parquet "
            + f"--input-mlpf-parquet {output_dir}/{corr_sample}_mlpf.parquet "
            + f"--corrections-file {jec_file} "
            + f"--jet-type {jet_type} --sample-name {corr_sample}"
        )
        write_bash_script(corr_script_path, cmd)

        rules_content += f"""
rule {corr_id}:
    input:
        "{output_dir}/{corr_sample}_pf.parquet",
        "{output_dir}/{corr_sample}_mlpf.parquet"
    output:
        "{corr_sentinel}",
        "{jec_file}"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{corr_script_path} && touch {{output[0]}}"
"""

    # 3. Validation Plots
    for sample, paths in all_samples.items():
        if paths["is_data"]:
            dv = scen.get("data_validation", {})
            golden_json = resolve_path(dv.get("golden_json", ""), spec)
            lumi_csv = resolve_path(dv.get("lumi_csv", ""), spec)

            # Data validation plots (CMSSW style)
            data_plot_id = f"data_plots_{sample}"
            data_plot_script_path = f"{jobs_dir}/plots/{data_plot_id}.sh"
            data_plot_sentinel = f"{jobs_dir}/plots/{data_plot_id}.done"

            # We use AK4 JECs by default for this validation
            jec_file = f"{output_dir}/jec_ak4_{corr_sample}.npz"

            cmd = (
                "PYTHONPATH=. python3 mlpf/plotting/cmssw_validation_data.py "
                + f"--input-pf '{paths['pf_dir']}/step4_NANO_*.root' "
                + f"--input-mlpf '{paths['mlpf_dir']}/step4_NANO_*.root' "
                + f"--golden-json {golden_json} --jec-file {jec_file} "
                + f"--lumi-csv {lumi_csv} --output-dir {output_dir}/cmssw_{sample}"
            )
            write_bash_script(data_plot_script_path, cmd)

            final_targets.append(data_plot_sentinel)
            rules_content += f"""
rule {data_plot_id}:
    input:
        "{jec_file}"
    output:
        "{data_plot_sentinel}"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{data_plot_script_path} && touch {{output}}"
"""
        else:
            # MET plots
            met_id = f"met_plots_{sample}"
            met_script_path = f"{jobs_dir}/plots/{met_id}.sh"
            met_sentinel = f"{jobs_dir}/plots/{met_id}.done"

            cmd = (
                "PYTHONPATH=. python3 mlpf/plotting/plot_met_validation.py "
                + f"--input-pf-parquet {output_dir}/{sample}_pf.parquet "
                + f"--input-mlpf-parquet {output_dir}/{sample}_mlpf.parquet "
                + f"--output-dir {output_dir} --sample-name {sample} --tev {scen['tev']}"
            )
            write_bash_script(met_script_path, cmd)

            final_targets.append(met_sentinel)
            rules_content += f"""
rule {met_id}:
    input:
        "{output_dir}/{sample}_pf.parquet",
        "{output_dir}/{sample}_mlpf.parquet"
    output:
        "{met_sentinel}"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{met_script_path} && touch {{output}}"
"""

            # Jet plots (only for MC samples or if desired for data - here we assume MC mostly)
            if not paths["is_data"]:
                for jet_type in scen["jet_types"]:
                    jec_file = f"{output_dir}/jec_{jet_type}_{corr_sample}.npz"
                    for fcut in scen["fiducial_cuts"]:
                        plot_id = f"jet_plots_{sample}_{jet_type}_{fcut}"
                        plot_script_path = f"{jobs_dir}/plots/{plot_id}.sh"
                        plot_sentinel = f"{jobs_dir}/plots/{plot_id}.done"

                        cmd = (
                            "PYTHONPATH=. python3 mlpf/plotting/plot_validation.py "
                            + f"--input-pf-parquet {output_dir}/{sample}_pf.parquet "
                            + f"--input-mlpf-parquet {output_dir}/{sample}_mlpf.parquet "
                            + f"--corrections-file {jec_file} "
                            + f"--output-dir {output_dir} --jet-type {jet_type} "
                            + f"--sample-name {sample} --fiducial-cuts {fcut} --tev {scen['tev']}"
                        )
                        write_bash_script(plot_script_path, cmd)

                        final_targets.append(plot_sentinel)
                        rules_content += f"""
rule {plot_id}:
    input:
        "{output_dir}/{sample}_pf.parquet",
        "{output_dir}/{sample}_mlpf.parquet",
        "{jec_file}"
    output:
        "{plot_sentinel}"
    resources:
        {get_resource_str(executor, mem_mb, partition, runtime, slurm_account=slurm_account)}
    container:
        "{container_img}"
    shell:
        "{plot_script_path} && touch {{output}}"
"""

    # Finalize Snakefile
    def fmt_list(lst):
        return "[" + ", ".join([f'"{x}"' for x in lst]) + "]"

    snakefile_content = "import os\n\n"
    snakefile_content += 'os.environ["GOTO_NUM_THREADS"]="1"\n'
    snakefile_content += 'os.environ["MKL_NUM_THREADS"]="1"\n'
    snakefile_content += 'os.environ["NUMEXPR_NUM_THREADS"]="1"\n'
    snakefile_content += 'os.environ["OMP_NUM_THREADS"]="1"\n'
    snakefile_content += 'os.environ["OPENBLAS_NUM_THREADS"]="1"\n'
    snakefile_content += 'os.environ["VECLIB_MAXIMUM_THREADS"]="1"\n\n'
    snakefile_content += "rule all:\n    input:\n"
    snakefile_content += "        " + fmt_list(final_targets) + "\n"
    snakefile_content += rules_content

    snakefile_path = f"{jobs_dir}/Snakefile"
    with open(snakefile_path, "w") as f:
        f.write(snakefile_content)

    print(f"Generated Snakemake workflow in {snakefile_path}")
    print(f"Generated {len(final_targets)} target plots.")

    # Write jobs_dir to a file for pixi consumption
    with open(".last_jobs_dir", "w") as f:
        f.write(jobs_dir)


if __name__ == "__main__":
    main()
