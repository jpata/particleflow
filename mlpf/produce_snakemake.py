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


def get_resource_str(executor, mem, partition, runtime, threads=1, gpus=0, gpu_type=None, mem_per_gpu=0, slurm_account=None, tmpdir=None):
    res = {}
    runtime_m = parse_runtime(runtime)
    if tmpdir:
        res["tmpdir"] = f'"{tmpdir}"'
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
    parser.add_argument("--ignore-failures", action="store_true", help="Ignore failures in gen/post steps")
    args = parser.parse_args()

    spec = load_spec(SPEC_FILE)

    if "cms" in args.production:
        req_steps = ["gen", "post", "tfds", "train"]
    elif "cld" in args.production or "clic" in args.production:
        req_steps = ["gen", "post", "tfds", "tfds_hit", "train"]
    else:
        req_steps = ["gen", "post", "tfds", "train"]

    project_root = resolve_path("${project.paths.project_root}", spec)

    # Target specific production
    if args.production not in spec["productions"]:
        raise ValueError(f"Production {args.production} not found in {SPEC_FILE}")

    prod_config = spec["productions"][args.production]
    prod_type = prod_config.get("type", "cms")

    # Determine models to train
    models_to_train = []
    # Resolve production type to match with model datasets
    # e.g. "cms_run3" -> "cms", "cld" -> "cld", "clic" -> "clic"
    prod_key = args.production.split("_")[0] if "cms" in args.production else args.production
    for mname, mspec in spec.get("models", {}).items():
        if mname == "defaults":
            continue
        m_dataset = mspec.get("dataset", "")
        if m_dataset == prod_key or m_dataset.startswith(prod_key + "_"):
            models_to_train.append(mname)

    if not models_to_train:
        # Fallback to the model specified in the production config if any
        if "model" in prod_config:
            models_to_train = [prod_config["model"]]

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

    # Resolve paths and directories
    scratch_root = resolve_path(spec["project"]["paths"]["scratch_root"], spec)
    tmpdir = resolve_path(spec["project"]["paths"].get("tmpdir", "/tmp"), spec)
    os.makedirs(tmpdir, exist_ok=True)

    # Apptainer/Singularity configuration
    main_container_img = spec["project"].get("container")
    gen_container_img = prod_config.get("gen_container", main_container_img)

    bind_mounts = spec["project"].get("bind_mounts", [])
    bind_args = "--writable-tmpfs"
    for bm in bind_mounts:
        bind_args += f" -B {bm}"
    if tmpdir and not any(bm.split(":")[0] == tmpdir for bm in bind_mounts):
        bind_args += f" -B {tmpdir}:/tmp"

    # Get postprocessing script from spec
    postproc_script = prod_config["postprocessing"]["script"]
    postproc_extra_args = prod_config["postprocessing"].get("args", {})

    config_dir = resolve_path(prod_config.get("config_dir", ""), spec)

    samples = prod_config["samples"]
    tfds_mappings = prod_config.get("tfds_mapping", {})
    tfds_hit_mappings = prod_config.get("tfds_hit_mapping", {})

    # Update jobs dir to include production name to avoid conflicts
    jobs_dir = f"{LOCAL_JOBS_DIR}/{args.production}"

    ensure_dir(f"{jobs_dir}/gen")
    ensure_dir(f"{jobs_dir}/post")
    ensure_dir(f"{jobs_dir}/tfds")
    ensure_dir(f"{jobs_dir}/tfds_hit")
    ensure_dir(f"{jobs_dir}/val")
    ensure_dir(tfds_root_dir)

    val_config = prod_config.get("validation", {})
    val_job_types = val_config.get("job_types", [])
    val_use_cuda = val_config.get("use_cuda", False)
    val_threads = val_config.get("threads", 1)
    mem_val = memory_config.get("val", 8000)

    step_data = {
        "gen": {"targets": [], "rules": ""},
        "post": {"targets": [], "rules": ""},
        "tfds": {"targets": [], "rules": ""},
        "tfds_hit": {"targets": [], "rules": ""},
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

        # 1. Generation Prototype Script
        gen_proto_path = f"{jobs_dir}/gen/gen_{sample_key}.sh"
        if prod_type == "cms":
            root_file_proto = os.path.join(sample_gen_root_dir, "pfntuple_${seed}.root")
            gen_base_dir = os.path.join(workspace_dir, "gen", output_subdir)
        elif prod_type == "key4hep":
            root_file_proto = os.path.join(sample_gen_root_dir, f"reco_{process_name}_" + "${seed}.root")
            gen_base_dir = os.path.join(workspace_dir, "gen")

        gen_cmd = f"bash {gen_script} {process_name} $seed {pu_type}"
        if copy_step2:
            gen_cmd += " True"
        if args.ignore_failures:
            gen_cmd += " || echo 'WARNING: Generation failed'"

        exports = (
            f"export OUTDIR={gen_base_dir}/"
            + f" && export CONFIG_DIR={config_dir}"
            + (f" && export CMSSWDIR={cmssw_dir}" if cmssw_dir else "")
            + f" && export WORKDIR={scratch_root}/{process_name}_$seed"
            + f" && export NEV={events_per_job}"
        )

        gen_proto_content = f"""
start_seed=$1
for (( i=0; i<{CHUNK_SIZE}; i++ )); do
    seed=$((start_seed + i))
    if [ ! -f {root_file_proto} ]; then
        echo "Generating {root_file_proto}"
        {exports}
        {gen_cmd}
        echo "Validating {root_file_proto}"
        python3 -c "import uproot; uproot.open('{root_file_proto}')"
    else
        echo "Skipping {root_file_proto}, already exists"
    fi
done
"""
        write_bash_script(gen_proto_path, gen_proto_content, project_root=project_root, tmpdir=tmpdir)

        # 1.5 Validation Prototype Script
        if copy_step2 and "val" in req_steps and prod_type == "cms":
            for job_type in val_job_types:
                val_proto_path = f"{jobs_dir}/val/val_{sample_key}_{job_type}.sh"
                val_cmd = (
                    f"WORKSPACE_DIR={workspace_dir} OUTPUT_SUBDIR={output_subdir} {f'CMSSWDIR={cmssw_dir} ' if cmssw_dir else ''}"
                    + f"NTHREADS={val_threads} bash mlpf/data/cms/valjob.sh {process_name} $seed {job_type} {val_use_cuda}"
                )
                val_proto_content = f"""
start_seed=$1
for (( i=0; i<{CHUNK_SIZE}; i++ )); do
    seed=$((start_seed + i))
    {val_cmd}
done
"""
                write_bash_script(val_proto_path, val_proto_content, project_root=project_root, tmpdir=tmpdir)

        # 2. Postprocessing Prototype Script
        post_proto_path = f"{jobs_dir}/post/post_{sample_key}.sh"
        if prod_type == "cms":
            root_file_proto = os.path.join(sample_gen_root_dir, "pfntuple_${seed}.root")
            post_file_final_proto = os.path.join(sample_post_dir, "pfntuple_${seed}.pkl.bz2")
            post_file_inter_proto = os.path.join(sample_post_dir, "pfntuple_${seed}.pkl")
        elif prod_type == "key4hep":
            root_file_proto = os.path.join(sample_gen_root_dir, f"reco_{process_name}_" + "${seed}.root")
            post_file_final_proto = os.path.join(sample_post_dir, f"reco_{process_name}_" + "${seed}.parquet")
            post_file_inter_proto = post_file_final_proto

        args_str = f"--input {root_file_proto} --outpath {sample_post_dir}"
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
            post_cmd = f"""
    if [ ! -f {post_file_final_proto} ]; then
        if [ -f {root_file_proto} ]; then
            echo "Postprocessing {root_file_proto}"
            {postproc_cmd}
            if [ -f {post_file_inter_proto} ]; then
                bzip2 -z {post_file_inter_proto}
            else
                echo "Error: Postprocessing failed to produce {post_file_inter_proto}"
                {exit_cmd}
            fi
        else
            echo "Error: Input file {root_file_proto} missing for postprocessing"
            {exit_cmd}
        fi
    else
        echo "Skipping {post_file_final_proto}, already exists"
    fi
"""
        else:  # key4hep / parquet
            post_cmd = f"""
    if [ ! -f {post_file_final_proto} ]; then
        if [ -f {root_file_proto} ]; then
            echo "Postprocessing {root_file_proto}"
            {postproc_cmd}
            if [ -f {post_file_final_proto} ]; then
                python3 -c "import awkward as ak; ak.from_parquet('{post_file_final_proto}')"
            else
                echo "Error: Postprocessing failed to produce {post_file_final_proto}"
                {exit_cmd}
            fi
        else
            echo "Error: Input file {root_file_proto} missing for postprocessing"
            {exit_cmd}
        fi
    else
        echo "Skipping {post_file_final_proto}, already exists"
    fi
"""
        post_proto_content = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
start_seed=$1
for (( i=0; i<{CHUNK_SIZE}; i++ )); do
    seed=$((start_seed + i))
    {post_cmd}
done
"""
        write_bash_script(post_proto_path, post_proto_content, project_root=project_root, tmpdir=tmpdir)

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

        if "post" in req_steps and (sample_key in tfds_mappings or sample_key in tfds_hit_mappings):
            step_data["post"]["targets"].append(f'"{jobs_dir}/post/post_{sample_key}_all.done"')
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
    input:
        {", ".join(step_data["gen"]["targets"])}
    output:
        "{jobs_dir}/gen/all.done"
    shell:
        "touch {{output}}"

rule gen_task:
    output:
        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"
    resources:
        {get_resource_str(executor, mem_gen, cpu_partition, cpu_runtime, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/gen/gen_{{wildcards.sample}}.sh {{wildcards.seed}} && touch {{output}}"
"""

    if "val" in req_steps and prod_type == "cms":
        val_rule_input = f'\n    input:\n        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"' if "gen" in req_steps else ""
        step_data["val"][
            "rules"
        ] += f"""
rule val:
    input:
        {", ".join(step_data["val"]["targets"])}
    output:
        "{jobs_dir}/val/all.done"
    shell:
        "touch {{output}}"

rule val_task:{val_rule_input}
    output:
        "{jobs_dir}/val/val_{{sample}}_{{job_type}}_{{seed}}.done"
    threads: {val_threads}
    resources:
        {get_resource_str(executor, mem_val, cpu_partition, cpu_runtime, threads=val_threads, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/val/val_{{wildcards.sample}}_{{wildcards.job_type}}.sh {{wildcards.seed}} && touch {{output}}"
"""

    if "post" in req_steps:
        post_rule_input = f'\n    input:\n        "{jobs_dir}/gen/gen_{{sample}}_{{seed}}.done"' if "gen" in req_steps else ""
        step_data["post"][
            "rules"
        ] += f"""
rule post:
    input:
        {", ".join(step_data["post"]["targets"])}
    output:
        "{jobs_dir}/post/all.done"
    shell:
        "touch {{output}}"

rule post_task:{post_rule_input}
    output:
        "{jobs_dir}/post/post_{{sample}}_{{seed}}.done"
    resources:
        {get_resource_str(executor, mem_post, cpu_partition, cpu_runtime, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{main_container_img}"
    shell:
        "{jobs_dir}/post/post_{{wildcards.sample}}.sh {{wildcards.seed}} && touch {{output}}"
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
                val_data_proto_path = f"{jobs_dir}/val/val_data_{val_sample_key}_{job_type}.sh"
                val_data_cmd = (
                    f"WORKSPACE_DIR={workspace_dir} OUTPUT_SUBDIR={val_sample_data.get('output_subdir', val_sample_key)} {f'CMSSWDIR={cmssw_dir} ' if cmssw_dir else ''} "
                    + f"INPUT_FILELIST={config_dir}/{input_filelist} "
                    + f"NTHREADS={val_data_threads} bash mlpf/data/cms/valjob_data.sh {val_sample_key} $seed {job_type} {val_data_use_cuda}"
                )
                val_data_proto_content = f"""
start_seed=$1
for (( i=0; i<{CHUNK_SIZE}; i++ )); do
    seed=$((start_seed + i))
    {val_data_cmd}
done
"""
                write_bash_script(val_data_proto_path, val_data_proto_content, project_root=project_root, tmpdir=tmpdir)

        step_data["val_data"][
            "rules"
        ] += f"""
rule val_data:
    input:
        {", ".join(step_data["val_data"]["targets"])}
    output:
        "{jobs_dir}/val_data/all.done"
    shell:
        "touch {{output}}"

rule val_data_task:
    output:
        "{jobs_dir}/val/val_data_{{sample}}_{{job_type}}_{{seed}}.done"
    threads: {val_data_threads}
    resources:
        {get_resource_str(executor, mem_val, cpu_partition, cpu_runtime, threads=val_data_threads, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{gen_container_img}"
    shell:
        "{jobs_dir}/val/val_data_{{wildcards.sample}}_{{wildcards.job_type}}.sh {{wildcards.seed}} && touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # PART 2: TFDS Conversion
    # -------------------------------------------------------------------------
    tfds_sentinels = []
    tfds_hit_sentinels = []

    tfds_mappings_to_process = {
        "tfds": (prod_config.get("tfds_mapping", {}), tfds_sentinels),
        "tfds_hit": (prod_config.get("tfds_hit_mapping", {}), tfds_hit_sentinels),
    }

    for step_name, (tfds_mappings, sentinels_list) in tfds_mappings_to_process.items():
        for sample_key, mapping in tfds_mappings.items():
            if sample_key not in samples:
                print(f"Warning: {step_name} mapping found for {sample_key} but no sample definition.")
                continue

            sample_data = samples[sample_key]
            builder_path = mapping["builder_path"]
            config_ids = mapping.get("config_ids", [1])

            process_name = sample_data["process_name"]
            output_subdir = sample_data.get("output_subdir", process_name)

            # Determine manual_dir for TFDS
            if prod_type == "cms":
                # For CMS, data is in workspace/post/subdir/process
                # TFDS builder expects workspace/post/subdir (containing process folder)
                manual_dir = os.path.join(workspace_dir, "post", output_subdir)
            else:
                # For Key4Hep, data is in workspace/post/process
                # TFDS builder expects workspace/post (containing process folder)
                manual_dir = os.path.join(workspace_dir, "post")

            tfds_script_path = f"{jobs_dir}/{step_name}/{step_name}_{sample_key}.sh"
            version = mapping.get("version")

            cmd = f"""
config_id=$1
tfds_id={sample_key}_{step_name}_$config_id
job_scratch_dir={scratch_root}/tfds_tmp/$tfds_id

export PYTHONPATH=$(pwd):$PYTHONPATH
export KERAS_BACKEND=torch
hostname
{f'export TFDS_VERSION={version}' if version else ''}
env

echo "Building TFDS for {builder_path} config $config_id"
echo "Manual dir: {manual_dir}"
echo "Scratch dir: $job_scratch_dir"

mkdir -p $job_scratch_dir
cleanup() {{
    if [ ! -z "$job_scratch_dir" ] && [ "$job_scratch_dir" != "{scratch_root}" ]; then
        echo "Cleaning up scratch directory $job_scratch_dir"
        rm -Rf $job_scratch_dir
    fi
}}
trap cleanup EXIT
export TMPDIR=$job_scratch_dir
export TEMPDIR=$job_scratch_dir
export TEMP=$job_scratch_dir
export TMP=$job_scratch_dir
tfds build {builder_path} --config $config_id --data_dir $job_scratch_dir --manual_dir {manual_dir} --overwrite
echo "Copying from $job_scratch_dir to {tfds_root_dir}"
cp -r $job_scratch_dir/* {tfds_root_dir}/
"""
            write_bash_script(tfds_script_path, cmd, project_root=project_root, tmpdir=tmpdir)

            if step_name in req_steps:
                target_expand = f"expand('{jobs_dir}/{step_name}/{step_name}_{sample_key}_{step_name}_{{config}}.done', config={config_ids})"
                step_data[step_name]["targets"].append(target_expand)
                sentinels_list.append(target_expand)

        if step_name in req_steps:
            tfds_rule_input = ""
            if "post" in req_steps:
                tfds_rule_input = f'\n    input:\n        "{jobs_dir}/post/post_{{sample}}_all.done"'

            step_data[step_name][
                "rules"
            ] += f"""
rule {step_name}_task:{tfds_rule_input}
    output:
        "{jobs_dir}/{step_name}/{step_name}_{{sample}}_{step_name}_{{config}}.done"
    resources:
        {get_resource_str(executor, mem_tfds, cpu_partition, cpu_runtime, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{main_container_img}"
    shell:
        "{jobs_dir}/{step_name}/{step_name}_{{wildcards.sample}}.sh {{wildcards.config}} && touch {{output}}"
"""

    if tfds_sentinels and "tfds" in req_steps:
        step_data["tfds"][
            "rules"
        ] += f"""
rule tfds:
    input:
        {", ".join(tfds_sentinels)}
    output:
        "{jobs_dir}/tfds/all.done"
    shell:
        "touch {{output}}"
"""

    if tfds_hit_sentinels and "tfds_hit" in req_steps:
        step_data["tfds_hit"][
            "rules"
        ] += f"""
rule tfds_hit:
    input:
        {", ".join(tfds_hit_sentinels)}
    output:
        "{jobs_dir}/tfds_hit/all.done"
    shell:
        "touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # PART 3: Model Training
    # -------------------------------------------------------------------------
    if "train" in req_steps and models_to_train:
        ensure_dir(f"{jobs_dir}/train")
        for mname in models_to_train:
            model_spec = spec["models"][mname]
            model_defaults = spec["models"].get("defaults", {})
            gpu_threads = model_spec.get("threads", model_defaults.get("threads", 16))
            gpu_count = model_spec.get("gpus", model_defaults.get("gpus", 0))
            gpu_type = model_spec.get("gpu_type", model_defaults.get("gpu_type", ""))
            mem_per_gpu_mb = model_spec.get("mem_per_gpu_mb", model_defaults.get("mem_per_gpu_mb", 8000))
            gpu_partition = resolve_path(model_spec.get("slurm_partition", model_defaults.get("slurm_partition", "gpu")), spec)
            gpu_runtime = resolve_path(model_spec.get("slurm_runtime", model_defaults.get("slurm_runtime", "120m")), spec)

            exp_name = f"{mname}_{args.production}"
            train_script_path = f"{jobs_dir}/train/train_{exp_name}.sh"
            train_sentinel = f"{jobs_dir}/train/train_{exp_name}.done"
            train_cmd = (
                f"python3 mlpf/pipeline.py --spec-file {SPEC_FILE} --model-name {mname} --production-name {args.production} train --gpus {gpu_count}"
            )

            cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR={tfds_root_dir}
export KERAS_BACKEND=torch
export TORCH_COMPILE_DISABLE=1
env
nvidia-smi
{train_cmd}
"""
            write_bash_script(train_script_path, cmd, project_root=project_root, tmpdir=tmpdir)

            train_inputs = []
            model_dataset = model_spec.get("dataset", prod_config.get("type"))
            if "hits" in model_dataset:
                if "tfds_hit" in req_steps:
                    train_inputs.append(f'"{jobs_dir}/tfds_hit/all.done"')
            else:
                if "tfds" in req_steps:
                    train_inputs.append(f'"{jobs_dir}/tfds/all.done"')

            train_rule_input = ""
            if train_inputs:
                train_rule_input = f"\n    input:\n        {', '.join(train_inputs)}"

            # snakemake rule names cannot contain hyphens
            rule_model_name = mname.replace("-", "_")

            step_data["train"]["targets"].append(f'"{train_sentinel}"')
            step_data["train"][
                "rules"
            ] += f"""
rule train_{rule_model_name}:{train_rule_input}
    output:
        "{train_sentinel}"
    threads: {gpu_threads}
    resources:
        {get_resource_str(executor, mem_train, gpu_partition, gpu_runtime, threads=gpu_threads, gpus=gpu_count, gpu_type=gpu_type, mem_per_gpu=mem_per_gpu_mb, slurm_account=slurm_account, tmpdir=tmpdir)}
    container:
        "{main_container_img}"
    shell:
        "{train_script_path} && touch {{output}}"
"""

    # -------------------------------------------------------------------------
    # Finalize Snakefile(s)
    # -------------------------------------------------------------------------
    def write_snakefile(path, targets, rules, tmpdir=None):
        if not targets and not rules:
            return
        with open(path, "w") as f:
            f.write("import os\n\n")
            f.write('os.environ["GOTO_NUM_THREADS"]="1"\n')
            f.write('os.environ["MKL_NUM_THREADS"]="1"\n')
            f.write('os.environ["NUMEXPR_NUM_THREADS"]="1"\n')
            f.write('os.environ["OMP_NUM_THREADS"]="1"\n')
            f.write('os.environ["OPENBLAS_NUM_THREADS"]="1"\n')
            f.write('os.environ["VECLIB_MAXIMUM_THREADS"]="1"\n')
            f.write('os.environ["PYTHONNOUSERSITE"]="1"\n')
            if tmpdir:
                f.write(f'os.environ["TMPDIR"]="{tmpdir}"\n')
                f.write(f'os.environ["TEMPDIR"]="{tmpdir}"\n')
                f.write(f'os.environ["TEMP"]="{tmpdir}"\n')
                f.write(f'os.environ["TMP"]="{tmpdir}"\n')
                f.write(f'os.environ["APPTAINER_TMPDIR"]="{tmpdir}"\n')
                f.write(f'os.environ["APPTAINER_CACHEDIR"]="{tmpdir}"\n')
                f.write(f'os.environ["SINGULARITY_TMPDIR"]="{tmpdir}"\n')
                f.write(f'os.environ["SINGULARITY_CACHEDIR"]="{tmpdir}"\n')
            f.write("\n")
            f.write("rule all:\n    input:\n        " + ",\n        ".join(targets) + "\n")
            f.write(rules)

    # Write individual Snakefiles for each step
    for step, data in step_data.items():
        if data["targets"] or data["rules"]:
            write_snakefile(f"{jobs_dir}/Snakefile_{step}", data["targets"], data["rules"], tmpdir=tmpdir)

    print(f"Generated Snakemake workflow in {jobs_dir}")
    print(f'Run with: snakemake --snakefile {jobs_dir}/Snakefile_STEP --cores 1 --use-apptainer --apptainer-args "{bind_args} --nv"')


if __name__ == "__main__":
    main()
