import os
import stat
import argparse
from mlpf.utils import load_spec, resolve_path

# Configuration
CHUNK_SIZE = 1
LOCAL_JOBS_DIR = "snakemake_jobs"
SPEC_FILE = "particleflow_spec.yaml"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_bash_script(path, content, project_root=None):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
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


def get_resource_str(executor, mem, partition, runtime, threads=1, gpus=0, gpu_type=None, mem_per_gpu=0, slurm_account=None):
    res = {}
    runtime_m = parse_runtime(runtime)
    if executor == "slurm":
        res["mem_mb"] = mem
        if gpus > 0 and mem_per_gpu > 0:
            res["mem_per_gpu"] = mem_per_gpu
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
        res["threads"] = threads
    elif executor == "condor":
        res["mem_mb"] = mem
        res["job_flavour"] = f'"{partition}"'
        res["runtime"] = runtime_m
        res["getenv"] = True
        if gpus > 0:
            res["request_gpus"] = gpus
    else:
        res["mem_mb"] = mem

    return ", ".join([f"{k}={v}" for k, v in res.items()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=str, default="cms_2025_main", help="Production name from spec file")
    parser.add_argument("--model", type=str, default=None, help="Model name from spec file to train")
    parser.add_argument("--ignore-failures", action="store_true", help="Ignore failures in gen/post steps")
    parser.add_argument("--steps", type=str, default="gen,post,tfds,train", help="Comma-separated steps to run: gen,post,tfds,train")
    parser.add_argument("--split", action="store_true", help="Split Snakefile into step-specific files")
    args = parser.parse_args()

    req_steps = args.steps.split(",")

    spec = load_spec(SPEC_FILE)
    project_root = resolve_path("${project.paths.project_root}", spec)

    # Target specific production
    if args.production not in spec["productions"]:
        raise ValueError(f"Production {args.production} not found in {SPEC_FILE}")

    prod_config = spec["productions"][args.production]
    prod_type = prod_config.get("type", "cms")

    if not args.model:
        args.model = prod_config.get("model", "pyg-cms-v1")

    executor = spec["project"].get("executor", "slurm")
    slurm_account = spec["project"].get("slurm_account")

    cmssw_dir = resolve_path(prod_config.get("environment", {}).get("cmssw_dir", ""), spec)

    cpu_partition = resolve_path(prod_config.get("slurm_partition", "main"), spec)
    cpu_runtime = resolve_path(prod_config.get("slurm_runtime", "120m"), spec)

    memory_config = prod_config.get("memory", {})
    mem_gen = memory_config.get("gen", 2000)
    mem_post = memory_config.get("post", 2000)
    mem_tfds = memory_config.get("tfds", 4000)
    mem_train = memory_config.get("train", 8000)

    # Resolve workspace dir and TFDS dir
    workspace_dir = resolve_path(prod_config["workspace_dir"], spec)
    # Unify TFDS output directory to be within the workspace
    tfds_root_dir = os.path.join(workspace_dir, "tfds")

    # Apptainer/Singularity configuration
    main_container_img = spec["project"].get("container")
    gen_container_img = prod_config.get("gen_container", main_container_img)

    bind_mounts = spec["project"].get("bind_mounts", [])
    bind_args = ""
    for bm in bind_mounts:
        bind_args += f" -B {bm}"

    # Get postprocessing script from spec
    postproc_script = prod_config["postprocessing"]["script"]
    postproc_extra_args = prod_config["postprocessing"].get("args", {})

    config_dir = resolve_path(prod_config.get("config_dir", ""), spec)

    scratch_root = resolve_path(spec["project"]["paths"]["scratch_root"], spec)

    samples = prod_config["samples"]
    tfds_mappings = prod_config.get("tfds_mapping", {})

    # Update jobs dir to include production name to avoid conflicts
    jobs_dir = f"{LOCAL_JOBS_DIR}/{args.production}"

    ensure_dir(f"{jobs_dir}/gen")
    ensure_dir(f"{jobs_dir}/post")
    ensure_dir(f"{jobs_dir}/tfds")
    ensure_dir(f"{jobs_dir}/val")
    ensure_dir(tfds_root_dir)

    val_config = prod_config.get("validation", {})
    val_job_types = val_config.get("job_types", [])
    val_use_cuda = val_config.get("use_cuda", False)
    val_threads = val_config.get("threads", 1)
    mem_val = memory_config.get("val", 8000)

    # Data structure for split snakefiles
    step_data = {
        "gen": {"targets": [], "rules": ""},
        "post": {"targets": [], "rules": ""},
        "tfds": {"targets": [], "rules": ""},
        "train": {"targets": [], "rules": ""},
        "val": {"targets": [], "rules": ""},
        "val_data": {"targets": [], "rules": ""},
    }

    # -------------------------------------------------------------------------
    # PART 1: Generation & Postprocessing (Per Chunk)
    # -------------------------------------------------------------------------
    for sample_key, sample_data in samples.items():
        process_name = sample_data["process_name"]
        seed_start, seed_end = sample_data["seed_range"]
        gen_script = sample_data["gen_script"]
        output_subdir = sample_data.get("output_subdir", process_name)
        events_per_job = sample_data.get("events_per_job", 100)
        pu_type = sample_data.get("pu_type", "nopu")
        copy_step2 = sample_data.get("copy_step2", False)

        # Unified Directory Structure
        if prod_type == "cms":
            # CMS: workspace/gen/subdir/process
            sample_gen_dir = os.path.join(workspace_dir, "gen", output_subdir, process_name)
            sample_post_dir = os.path.join(workspace_dir, "post", output_subdir, process_name)
        else:
            # Key4Hep/Simple: workspace/gen/process
            sample_gen_dir = os.path.join(workspace_dir, "gen", process_name)
            sample_post_dir = os.path.join(workspace_dir, "post", process_name)

        # Convention: generated files go into a 'root' subdirectory
        sample_gen_root_dir = os.path.join(sample_gen_dir, "root")

        ensure_dir(sample_gen_dir)
        ensure_dir(sample_gen_root_dir)
        ensure_dir(sample_post_dir)

        # Iterate in chunks to write scripts
        for chunk_start in range(seed_start, seed_end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, seed_end)
            chunk_id = f"{sample_key}_{chunk_start}"

            # 1. Generation Script
            gen_script_path = f"{jobs_dir}/gen/gen_{chunk_id}.sh"
            gen_cmd_lines = []

            for seed in range(chunk_start, chunk_end):
                if prod_type == "cms":
                    gen_base_dir = os.path.join(workspace_dir, "gen", output_subdir)
                    root_file = os.path.join(sample_gen_root_dir, f"pfntuple_{seed}.root")
                elif prod_type == "key4hep":
                    gen_base_dir = os.path.join(workspace_dir, "gen")
                    root_file = os.path.join(sample_gen_root_dir, f"reco_{process_name}_{seed}.root")

                exports = (
                    f"export OUTDIR={gen_base_dir}/"
                    + f" && export CONFIG_DIR={config_dir}"
                    + (f" && export CMSSWDIR={cmssw_dir}" if cmssw_dir else "")
                    + f" && export WORKDIR={scratch_root}/{process_name}_{seed}"
                    + f" && export NEV={events_per_job}"
                )
                gen_cmd = f"bash {gen_script} {process_name} {seed} {pu_type}"
                if copy_step2:
                    gen_cmd += " True"

                if args.ignore_failures:
                    gen_cmd += " || echo 'WARNING: Generation failed'"

                cmd = f"""
if [ ! -f {root_file} ]; then
    echo "Generating {root_file}"
    {exports}
    {gen_cmd}
else
    echo "Skipping {root_file}, already exists"
fi
"""
                gen_cmd_lines.append(cmd)

            write_bash_script(gen_script_path, "\n".join(gen_cmd_lines), project_root=project_root)

            # 1.5 Validation scripts
            if copy_step2 and "val" in req_steps and prod_type == "cms":
                for job_type in val_job_types:
                    val_id = f"{sample_key}_{job_type}_{chunk_start}"
                    val_script_path = f"{jobs_dir}/val/val_{val_id}.sh"
                    val_cmd_lines = []
                    for seed in range(chunk_start, chunk_end):
                        val_cmd = (
                            f"WORKSPACE_DIR={workspace_dir} OUTPUT_SUBDIR={output_subdir} {f'CMSSWDIR={cmssw_dir} ' if cmssw_dir else ''}"
                            + f"NTHREADS={val_threads} bash mlpf/data/cms/valjob.sh {process_name} {seed} {job_type} {val_use_cuda}"
                        )
                        val_cmd_lines.append(val_cmd)
                    write_bash_script(val_script_path, "\n".join(val_cmd_lines), project_root=project_root)

            # 2. Postprocessing Script
            post_script_path = f"{jobs_dir}/post/post_{chunk_id}.sh"
            post_cmd_lines = []
            for seed in range(chunk_start, chunk_end):
                if prod_type == "cms":
                    root_file = os.path.join(sample_gen_root_dir, f"pfntuple_{seed}.root")
                    post_file_final = os.path.join(sample_post_dir, f"pfntuple_{seed}.pkl.bz2")
                    post_file_inter = os.path.join(sample_post_dir, f"pfntuple_{seed}.pkl")
                elif prod_type == "key4hep":
                    root_file = os.path.join(sample_gen_root_dir, f"reco_{process_name}_{seed}.root")
                    post_file_final = os.path.join(sample_post_dir, f"reco_{process_name}_{seed}.parquet")
                    post_file_inter = post_file_final

                args_str = f"--input {root_file} --outpath {sample_post_dir}"
                for k, v in postproc_extra_args.items():
                    if isinstance(v, bool):
                        if v:
                            args_str += f" --{k}"
                    else:
                        args_str += f" --{k} {v}"

                postproc_cmd = f"python3 {postproc_script} {args_str}"
                exit_cmd = "exit 1"
                if args.ignore_failures:
                    postproc_cmd += " || echo 'WARNING: Postprocessing failed'"
                    exit_cmd = "echo 'Ignoring failure'; true"

                if prod_type == "cms":
                    cmd = f"""
if [ ! -f {post_file_final} ]; then
    if [ -f {root_file} ]; then
        echo "Postprocessing {root_file}"
        {postproc_cmd}
        if [ -f {post_file_inter} ]; then
            bzip2 -z {post_file_inter}
        else
            echo "Error: Postprocessing failed to produce {post_file_inter}"
            {exit_cmd}
        fi
    else
        echo "Error: Input file {root_file} missing for postprocessing"
        {exit_cmd}
    fi
else
    echo "Skipping {post_file_final}, already exists"
fi
"""
                else:  # key4hep / parquet
                    cmd = f"""
if [ ! -f {post_file_final} ]; then
    if [ -f {root_file} ]; then
        echo "Postprocessing {root_file}"
        {postproc_cmd}
    else
        echo "Error: Input file {root_file} missing for postprocessing"
        {exit_cmd}
    fi
else
    echo "Skipping {post_file_final}, already exists"
fi
"""
                post_cmd_lines.append(cmd)
            write_bash_script(post_script_path, "\n".join(post_cmd_lines), project_root=project_root)

        # Add targets to rule all for this sample
        if "gen" in req_steps:
            step_data["gen"]["targets"].append(
                f"expand('{jobs_dir}/gen/gen_{sample_key}_{{seed}}.done', seed=range({seed_start}, {seed_end}, {CHUNK_SIZE}))"
            )

        if copy_step2 and "val" in req_steps and prod_type == "cms":
            for job_type in val_job_types:
                step_data["val"]["targets"].append(
                    f"expand('{jobs_dir}/val/val_{sample_key}_{job_type}_{{seed}}.done', seed=range({seed_start}, {seed_end}, {CHUNK_SIZE}))"
                )

        if "post" in req_steps and (sample_key in tfds_mappings):
            step_data["post"]["targets"].append(
                f"expand('{jobs_dir}/post/post_{sample_key}_{{seed}}.done', seed=range({seed_start}, {seed_end}, {CHUNK_SIZE}))"
            )

        # Grouping rule for TFDS input
        if "post" in req_steps and sample_key in tfds_mappings:
            step_data["post"][
                "rules"
            ] += f"""
rule post_{sample_key}_all:
    input:
        expand("{jobs_dir}/post/post_{sample_key}_{{seed}}.done", seed=range({seed_start}, {seed_end}, {CHUNK_SIZE}))
    output:
        "{jobs_dir}/post/post_{sample_key}_all.done"
    shell:
        "touch {{output}}"
"""

    # Generic rules for per-chunk tasks
    if "gen" in req_steps:
        step_data["gen"][
            "rules"
        ] += f"""
rule gen:
    output:
        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"
    resources:
        {get_resource_str(executor, mem_gen, cpu_partition, cpu_runtime, slurm_account=slurm_account)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/gen/gen_{{wildcards.sample}}_{{wildcards.seed}}.sh && touch {{output}}"
"""

    if "val" in req_steps and prod_type == "cms":
        val_rule_input = f'\n    input:\n        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"' if "gen" in req_steps else ""
        step_data["val"][
            "rules"
        ] += f"""
rule val:{val_rule_input}
    output:
        "{jobs_dir}/val/val_{{sample}}_{{job_type}}_{{seed}}.done"
    threads: {val_threads}
    resources:
        {get_resource_str(executor, mem_val, cpu_partition, cpu_runtime, threads=val_threads, slurm_account=slurm_account)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/val/val_{{wildcards.sample}}_{{wildcards.job_type}}_{{wildcards.seed}}.sh && touch {{output}}"
"""

    if "post" in req_steps:
        post_rule_input = f'\n    input:\n        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"' if "gen" in req_steps else ""
        step_data["post"][
            "rules"
        ] += f"""
rule post:{post_rule_input}
    output:
        "{jobs_dir}/post/post_{{sample}}_{{seed}}.done"
    resources:
        {get_resource_str(executor, mem_post, cpu_partition, cpu_runtime, slurm_account=slurm_account)}
    container:
        "{main_container_img}"
    shell:
        "{jobs_dir}/post/post_{{wildcards.sample}}_{{wildcards.seed}}.sh && touch {{output}}"
"""

    # 1.6 Data Validation
    if "val_data" in req_steps and prod_type == "cms":
        val_data_config = prod_config.get("validation_data", {})
        val_data_job_types = val_data_config.get("job_types", [])
        val_data_use_cuda = val_data_config.get("use_cuda", False)
        val_data_threads = val_data_config.get("threads", 1)
        val_data_samples = val_data_config.get("samples", {})

        for val_sample_key, val_sample_data in val_data_samples.items():
            input_filelist = resolve_path(val_sample_data["input_filelist"], spec)
            seed_start, seed_end = val_sample_data["seed_range"]
            step_data["val_data"]["targets"].append(
                f"expand('{jobs_dir}/val/val_data_{val_sample_key}_{{job_type}}_{{seed}}.done', job_type={val_data_job_types}, seed=range({seed_start}, {seed_end}, {CHUNK_SIZE}))"
            )

            for job_type in val_data_job_types:
                for chunk_start in range(seed_start, seed_end, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, seed_end)
                    val_id = f"{val_sample_key}_{job_type}_{chunk_start}"
                    val_script_path = f"{jobs_dir}/val/val_data_{val_id}.sh"
                    val_cmd_lines = []
                    for seed in range(chunk_start, chunk_end):
                        val_cmd = (
                            f"WORKSPACE_DIR={workspace_dir} OUTPUT_SUBDIR={val_sample_data.get('output_subdir', val_sample_key)} {f'CMSSWDIR={cmssw_dir} ' if cmssw_dir else ''} "
                            + f"INPUT_FILELIST={config_dir}/{input_filelist} "
                            + f"NTHREADS={val_data_threads} bash mlpf/data/cms/valjob_data.sh {val_sample_key} {seed} {job_type} {val_data_use_cuda}"
                        )
                        val_cmd_lines.append(val_cmd)
                    write_bash_script(val_script_path, "\n".join(val_cmd_lines), project_root=project_root)

        step_data["val_data"][
            "rules"
        ] += f"""
rule val_data:
    output:
        "{jobs_dir}/val/val_data_{{sample}}_{{job_type}}_{{seed}}.done"
    threads: {val_data_threads}
    resources:
        {get_resource_str(executor, mem_val, cpu_partition, cpu_runtime, threads=val_data_threads, slurm_account=slurm_account)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/val/val_data_{{wildcards.sample}}_{{wildcards.job_type}}_{{wildcards.seed}}.sh && touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # PART 2: TFDS Conversion
    # -------------------------------------------------------------------------
    tfds_sentinels = []
    for sample_key, mapping in tfds_mappings.items():
        if sample_key not in samples:
            continue
        sample_data = samples[sample_key]
        builder_path = mapping["builder_path"]
        config_ids = mapping.get("config_ids", [1])
        output_subdir = sample_data.get("output_subdir", sample_data["process_name"])

        manual_dir = os.path.join(workspace_dir, "post", output_subdir) if prod_type == "cms" else os.path.join(workspace_dir, "post")

        for config_id in config_ids:
            tfds_id = f"{sample_key}_tfds_{config_id}"
            tfds_script_path = f"{jobs_dir}/tfds/tfds_{tfds_id}.sh"
            tfds_sentinel = f"{jobs_dir}/tfds/tfds_{tfds_id}.done"
            job_scratch_dir = os.path.join(scratch_root, "tfds_tmp", tfds_id)
            version = mapping.get("version")
            tfds_build_cmd = f"tfds build {builder_path} --config {config_id} --data_dir {job_scratch_dir} --manual_dir {manual_dir} --overwrite"

            cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
export KERAS_BACKEND=torch
hostname
{f'export TFDS_VERSION={version}' if version else ''}
mkdir -p {job_scratch_dir}
cleanup() {{
    if [ ! -z "{job_scratch_dir}" ] && [ "{job_scratch_dir}" != "{scratch_root}" ]; then
        rm -Rf {job_scratch_dir}
    fi
}}
trap cleanup EXIT
{tfds_build_cmd}
cp -r {job_scratch_dir}/* {tfds_root_dir}/
"""
            write_bash_script(tfds_script_path, cmd, project_root=project_root)
            tfds_sentinels.append(tfds_sentinel)

            if "tfds" in req_steps:
                step_data["tfds"]["targets"].append(f'"{tfds_sentinel}"')
                tfds_rule_input = f'\n    input:\n        "{jobs_dir}/post/{sample_key}_all.done"' if "post" in req_steps else ""
                step_data["tfds"][
                    "rules"
                ] += f"""
rule tfds_{tfds_id}:{tfds_rule_input}
    output:
        "{tfds_sentinel}"
    resources:
        {get_resource_str(executor, mem_tfds, cpu_partition, cpu_runtime, slurm_account=slurm_account)}
    container:
        "{main_container_img}"
    shell:
        "{tfds_script_path} && touch {{output}}"
"""

    if tfds_sentinels and "tfds" in req_steps:
        step_data["tfds"][
            "rules"
        ] += f"""
rule tfds_all:
    input:
        {", ".join([f'"{s}"' for s in tfds_sentinels])}
    output:
        "{jobs_dir}/tfds/all.done"
    shell:
        "touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # PART 3: Model Training
    # -------------------------------------------------------------------------
    if args.model:
        ensure_dir(f"{jobs_dir}/train")
        model_spec = spec["models"][args.model]
        model_defaults = spec["models"].get("defaults", {})
        gpu_threads = model_spec.get("threads", model_defaults.get("threads", 16))
        gpu_count = model_spec.get("gpus", model_defaults.get("gpus", 0))
        gpu_type = model_spec.get("gpu_type", model_defaults.get("gpu_type", ""))
        mem_per_gpu_mb = model_spec.get("mem_per_gpu_mb", model_defaults.get("mem_per_gpu_mb", 8000))
        gpu_partition = resolve_path(model_spec.get("slurm_partition", model_defaults.get("slurm_partition", "gpu")), spec)
        gpu_runtime = resolve_path(model_spec.get("slurm_runtime", model_defaults.get("slurm_runtime", "120m")), spec)

        exp_name = f"{args.model}_{args.production}"
        train_script_path = f"{jobs_dir}/train/train_{exp_name}.sh"
        train_sentinel = f"{jobs_dir}/train/train_{exp_name}.done"
        train_cmd = (
            f"python3 mlpf/pipeline.py --spec-file {SPEC_FILE} --model-name {args.model} --production-name {args.production} train --gpus {gpu_count}"
        )

        cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR={tfds_root_dir}
export KERAS_BACKEND=torch
export TORCH_COMPILE_DISABLE=1
{train_cmd}
"""
        write_bash_script(train_script_path, cmd, project_root=project_root)

        if "train" in req_steps:
            step_data["train"]["targets"].append(f'"{train_sentinel}"')
            train_rule_input = f'\n    input:\n        "{jobs_dir}/tfds/all.done"' if "tfds" in req_steps else ""
            step_data["train"][
                "rules"
            ] += f"""
rule train_{args.model.replace("-", "_")}:{train_rule_input}
    output:
        "{train_sentinel}"
    threads: {gpu_threads}
    resources:
        {get_resource_str(executor, mem_train, gpu_partition, gpu_runtime, threads=gpu_threads, gpus=gpu_count, gpu_type=gpu_type, mem_per_gpu=mem_per_gpu_mb, slurm_account=slurm_account)}
    container:
        "{main_container_img}"
    shell:
        "{train_script_path} && touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # Finalize Snakefile(s)
    # -------------------------------------------------------------------------
    def write_snakefile(path, targets, rules):
        if not targets and not rules:
            return
        with open(path, "w") as f:
            f.write("import os\n\n")
            f.write('os.environ["GOTO_NUM_THREADS"]="1"\n')
            f.write('os.environ["MKL_NUM_THREADS"]="1"\n')
            f.write('os.environ["NUMEXPR_NUM_THREADS"]="1"\n')
            f.write('os.environ["OMP_NUM_THREADS"]="1"\n')
            f.write('os.environ["OPENBLAS_NUM_THREADS"]="1"\n')
            f.write('os.environ["VECLIB_MAXIMUM_THREADS"]="1"\n\n')
            f.write("rule all:\n    input:\n        " + ",\n        ".join(targets) + "\n")
            f.write(rules)

    if args.split:
        for step, data in step_data.items():
            if data["targets"] or data["rules"]:
                write_snakefile(f"{jobs_dir}/Snakefile.{step}", data["targets"], data["rules"])

        # Also write a Master Snakefile that includes all
        with open(f"{jobs_dir}/Snakefile", "w") as f:
            for step in step_data.keys():
                if step_data[step]["targets"] or step_data[step]["rules"]:
                    f.write(f'include: "Snakefile.{step}"\n')
    else:
        # Combine everything into one Snakefile
        all_targets = []
        all_rules = ""
        for step in step_data.values():
            all_targets.extend(step["targets"])
            all_rules += step["rules"]
        write_snakefile(f"{jobs_dir}/Snakefile", all_targets, all_rules)

    print(f"Generated Snakemake workflow in {jobs_dir}")
    print(f'Run with: snakemake --snakefile {jobs_dir}/Snakefile --cores 1 --use-apptainer --apptainer-args "{bind_args} --nv"')


if __name__ == "__main__":
    main()
