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


def write_bash_script(path, content):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=str, default="cms_2025_main", help="Production name from spec file")
    parser.add_argument("--model", type=str, help="Model name from spec file to train")
    parser.add_argument("--steps", type=str, default="gen,post,tfds,train", help="Comma-separated steps to run: gen,post,tfds,train")
    args = parser.parse_args()

    req_steps = args.steps.split(",")
    enable_gen = "gen" in req_steps
    enable_post = "post" in req_steps
    enable_tfds = "tfds" in req_steps
    enable_train = "train" in req_steps

    spec = load_spec(SPEC_FILE)

    # Target specific production
    if args.production not in spec["productions"]:
        raise ValueError(f"Production {args.production} not found in {SPEC_FILE}")

    prod_config = spec["productions"][args.production]
    prod_type = prod_config.get("type", "cms")

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

    config_dir = prod_config.get("config_dir", "")

    scratch_root = resolve_path(spec["project"]["paths"]["scratch_root"], spec)

    samples = prod_config["samples"]
    tfds_mappings = prod_config.get("tfds_mapping", {})

    # Update jobs dir to include production name to avoid conflicts
    jobs_dir = f"{LOCAL_JOBS_DIR}/{args.production}"

    ensure_dir(f"{jobs_dir}/gen")
    ensure_dir(f"{jobs_dir}/post")
    ensure_dir(f"{jobs_dir}/tfds")
    ensure_dir(tfds_root_dir)

    snakefile_content = "rule all:\n    input:\n"

    # Track completion files for all chunks of all samples
    all_sample_post_sentinels = {}

    # Track targets for rule all
    final_targets = []

    rules_content = ""

    # -------------------------------------------------------------------------
    # PART 1: Generation & Postprocessing (Per Chunk)
    # -------------------------------------------------------------------------
    for sample_key, sample_data in samples.items():
        process_name = sample_data["process_name"]
        seed_start, seed_end = sample_data["seed_range"]
        gen_script = sample_data["gen_script"]
        output_subdir = sample_data.get("output_subdir", process_name)
        events_per_job = sample_data.get("events_per_job", 100)

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

        sample_post_sentinels = []

        # Iterate in chunks
        for chunk_start in range(seed_start, seed_end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, seed_end)
            chunk_id = f"{sample_key}_{chunk_start}"

            # Sentinels
            gen_sentinel = f"{jobs_dir}/gen/gen_{chunk_id}.done"
            post_sentinel = f"{jobs_dir}/post/post_{chunk_id}.done"

            # 1. Generation
            if enable_gen:
                gen_script_path = f"{jobs_dir}/gen/gen_{chunk_id}.sh"
                gen_cmd_lines = []

                for seed in range(chunk_start, chunk_end):
                    # Prepare common variables
                    if prod_type == "cms":
                        gen_base_dir = os.path.join(workspace_dir, "gen", output_subdir)
                        root_file = os.path.join(sample_gen_root_dir, f"pfntuple_{seed}.root")
                    elif prod_type == "key4hep":
                        gen_base_dir = os.path.join(workspace_dir, "gen")
                        root_file = os.path.join(sample_gen_root_dir, f"reco_{process_name}_{seed}.root")

                    exports = (
                        f"export OUTDIR={gen_base_dir}/"
                        + f" && export CONFIG_DIR={config_dir}"
                        + f" && export WORKDIR={scratch_root}/{process_name}_{seed}"
                        + f" && export NEV={events_per_job}"
                    )
                    gen_cmd = f"bash {gen_script} {process_name} {seed}"

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

                write_bash_script(gen_script_path, "\n".join(gen_cmd_lines))

                rules_content += f"""
rule gen_{chunk_id}:
    output:
        "{gen_sentinel}"
    resources:
        mem_mb={mem_gen}
    container:
        "{gen_container_img}"
    shell:
        "{gen_script_path} && touch {{output}}"
"""
                if "gen" in req_steps:
                    final_targets.append(gen_sentinel)

            # 2. Postprocessing
            if enable_post:
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
            exit 1
        fi
    else
        echo "Error: Input file {root_file} missing for postprocessing"
        exit 1
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
        exit 1
    fi
else
    echo "Skipping {post_file_final}, already exists"
fi
"""

                    post_cmd_lines.append(cmd)

                write_bash_script(post_script_path, "\n".join(post_cmd_lines))

                sample_post_sentinels.append(post_sentinel)
                if "post" in req_steps:
                    final_targets.append(post_sentinel)

                rules_content += f"""
rule post_{chunk_id}:
    input:
        "{gen_sentinel}"
    output:
        "{post_sentinel}"
    resources:
        mem_mb={mem_post}
    container:
        "{main_container_img}"
    shell:
        "{post_script_path} && touch {{output}}"
"""

        all_sample_post_sentinels[sample_key] = sample_post_sentinels

    # -------------------------------------------------------------------------
    # PART 2: TFDS Conversion (Per Config ID / Split)
    # -------------------------------------------------------------------------
    tfds_sentinels = []

    if enable_tfds:
        for sample_key, mapping in tfds_mappings.items():
            if sample_key not in samples:
                print(f"Warning: TFDS mapping found for {sample_key} but no sample definition.")
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

            for config_id in config_ids:
                tfds_id = f"{sample_key}_tfds_{config_id}"
                tfds_script_path = f"{jobs_dir}/tfds/tfds_{tfds_id}.sh"
                tfds_sentinel = f"{jobs_dir}/tfds/tfds_{tfds_id}.done"

                version = mapping.get("version")
                tfds_build_cmd = f"tfds build {builder_path} --config {config_id} --data_dir {tfds_root_dir} --manual_dir {manual_dir} --overwrite"

                cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
{f'export TFDS_VERSION={version}' if version else ''}

echo "Building TFDS for {builder_path} config {config_id}"
echo "Manual dir: {manual_dir}"

{tfds_build_cmd}
"""
                write_bash_script(tfds_script_path, cmd)

                input_sentinels_str = ",\n        ".join([f'"{s}"' for s in all_sample_post_sentinels.get(sample_key, [])])

                rules_content += f"""
rule tfds_{tfds_id}:
    input:
        {input_sentinels_str}
    output:
        "{tfds_sentinel}"
    resources:
        mem_mb={mem_tfds}
    container:
        "{main_container_img}"
    shell:
        "{tfds_script_path} && touch {{output}}"
"""
                tfds_sentinels.append(tfds_sentinel)
                if "tfds" in req_steps:
                    final_targets.append(tfds_sentinel)

    # -------------------------------------------------------------------------
    # PART 3: Model Training
    # -------------------------------------------------------------------------
    if enable_train:
        if not args.model:
            raise ValueError("A model must be specified with --model for the 'train' step.")
        if args.model not in spec["models"]:
            raise ValueError(f"Model {args.model} not found in {SPEC_FILE}")

        ensure_dir(f"{jobs_dir}/train")

        model_spec = spec["models"][args.model]
        gpu_count = model_spec.get("gpus", 0)
        gpu_type = model_spec.get("gpu_type", "")
        mem_per_gpu_mb = model_spec.get("mem_per_gpu_mb", 0)

        exp_name = f"{args.model}_{args.production}"

        train_script_path = f"{jobs_dir}/train/train_{exp_name}.sh"
        train_sentinel = f"{jobs_dir}/train/train_{exp_name}.done"

        train_cmd = f"python3 mlpf/pipeline.py --spec-file {SPEC_FILE} --model-name {args.model} --production-name {args.production} train"

        cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR={tfds_root_dir}

{train_cmd}
"""
        write_bash_script(train_script_path, cmd)

        input_sentinels_str = " ,\n        ".join([f'"{s}"' for s in tfds_sentinels])

        # snakemake rule names cannot contain hyphens
        rule_model_name = args.model.replace("-", "_")

        # Constructing the resources string
        resources_str = f"mem_mb={mem_train}"
        if gpu_count > 0:
            if gpu_type:
                resources_str += f', gres="gpu:{gpu_type}:{gpu_count}"'
            else:
                resources_str += f", gpu={gpu_count}"
        if mem_per_gpu_mb > 0 and gpu_count > 0:
            resources_str += f', "mem-per-gpu"={mem_per_gpu_mb}'

        rules_content += f"""
rule train_{rule_model_name}:
    input:
        {input_sentinels_str}
    output:
        "{train_sentinel}"
    resources:
        {resources_str}
    container:
        "{main_container_img}"
    shell:
        "{train_script_path} && touch {{output}}"
"""
        if "train" in req_steps:
            final_targets.append(train_sentinel)

    # -------------------------------------------------------------------------
    # Finalize Snakefile
    # -------------------------------------------------------------------------

    def fmt_list(lst):
        return "[" + ", ".join([f'"{x}"' for x in lst]) + "]"

    snakefile_content += "        " + fmt_list(final_targets) + "\n"
    snakefile_content += rules_content

    snakefile_path = f"{jobs_dir}/Snakefile"
    with open(snakefile_path, "w") as f:
        f.write(snakefile_content)

    print(f"Generated Snakemake workflow in {snakefile_path}")
    print(f"Generated {len(final_targets)} target jobs.")
    print(f'Run with: snakemake --snakefile {snakefile_path} --cores 1 --use-apptainer --apptainer-args "{bind_args}"')


if __name__ == "__main__":
    main()
