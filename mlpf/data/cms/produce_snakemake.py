import os
import stat
import yaml
import re

# Configuration
CHUNK_SIZE = 100
LOCAL_JOBS_DIR = "mlpf/data/cms/snakemake_jobs"
SPEC_FILE = "particleflow_spec.yaml"

def load_spec(spec_file):
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    return spec

def resolve_path(path, spec):
    # Simple recursive substitution for ${...}
    def replace(match):
        key_path = match.group(1).split('.')
        val = spec
        for k in key_path:
            val = val.get(k)
            if val is None: return match.group(0) # fail gracefully
        return str(val)
    
    return re.sub(r'\$\{(.+?)\}', replace, path)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def write_bash_script(path, content):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

def main():
    spec = load_spec(SPEC_FILE)
    
    # Target specific production
    prod_config = spec['productions']['cms_2025_main']
    
    # Resolve workspace dir and TFDS dir
    workspace_dir = resolve_path(prod_config['workspace_dir'], spec)
    tfds_root_dir = resolve_path(spec['project']['paths']['tfds_dir'], spec)
    
    # Apptainer configuration
    container_img = spec['project'].get('container')
    bind_mounts = spec['project'].get('bind_mounts', [])
    bind_args = ""
    for bm in bind_mounts:
        bind_args += f" -B {bm}"
    
    # Get postprocessing script from spec
    postproc_script = prod_config['postprocessing']['script']

    samples = prod_config['samples']
    tfds_mappings = prod_config.get('tfds_mapping', {})

    ensure_dir(f"{LOCAL_JOBS_DIR}/gen")
    ensure_dir(f"{LOCAL_JOBS_DIR}/post")
    ensure_dir(f"{LOCAL_JOBS_DIR}/tfds")
    
    snakefile_content = "rule all:\n    input:\n"
    
    # Track completion files for all chunks of all samples
    all_sample_post_sentinels = {}
    
    rules_content = ""

    # -------------------------------------------------------------------------
    # PART 1: Generation & Postprocessing (Per Chunk)
    # -------------------------------------------------------------------------
    for sample_key, sample_data in samples.items():
        process_name = sample_data['process_name']
        seed_start, seed_end = sample_data['seed_range']
        gen_script = sample_data['gen_script']
        output_subdir = sample_data['output_subdir']

        # Full output path structure: {workspace_dir}/{subdir}/{process_name}
        sample_out_dir = os.path.join(workspace_dir, output_subdir, process_name)
        root_dir = os.path.join(sample_out_dir, "root")
        raw_dir = os.path.join(sample_out_dir, "raw")
        
        ensure_dir(root_dir)
        ensure_dir(raw_dir)

        sample_post_sentinels = []

        # Iterate in chunks
        for chunk_start in range(seed_start, seed_end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, seed_end)
            chunk_id = f"{sample_key}_{chunk_start}"

            # 1. Generation Script
            gen_script_path = f"{LOCAL_JOBS_DIR}/gen/gen_{chunk_id}.sh"
            gen_cmd_lines = []
            
            for seed in range(chunk_start, chunk_end):
                root_file = os.path.join(root_dir, f"pfntuple_{seed}.root")
                cmd = f"""
if [ ! -f {root_file} ]; then
    echo "Generating {root_file}"
    scripts/tallinn/cmssw-el8.sh mlpf/data/cms/{gen_script} {process_name} {seed}
else
    echo "Skipping {root_file}, already exists"
fi
"""
                gen_cmd_lines.append(cmd)
            
            write_bash_script(gen_script_path, "\n".join(gen_cmd_lines))

            # 2. Postprocessing Script
            post_script_path = f"{LOCAL_JOBS_DIR}/post/post_{chunk_id}.sh"
            post_cmd_lines = []
            
            for seed in range(chunk_start, chunk_end):
                root_file = os.path.join(root_dir, f"pfntuple_{seed}.root")
                pkl_file_bz2 = os.path.join(raw_dir, f"pfntuple_{seed}.pkl.bz2")
                pkl_file = os.path.join(raw_dir, f"pfntuple_{seed}.pkl")
                
                postproc_cmd = f"python3 {postproc_script} --input {root_file} --outpath {raw_dir}"
                if container_img:
                    postproc_cmd = f"apptainer exec {bind_args} {container_img} {postproc_cmd}"

                cmd = f"""
if [ ! -f {pkl_file_bz2} ]; then
    if [ -f {root_file} ]; then
        echo "Postprocessing {root_file}"
        {postproc_cmd}
        if [ -f {pkl_file} ]; then
            bzip2 -z {pkl_file}
        else
            echo "Error: Postprocessing failed to produce {pkl_file}"
            exit 1
        fi
    else
        echo "Error: Input file {root_file} missing for postprocessing"
        exit 1
    fi
else
    echo "Skipping {pkl_file_bz2}, already exists"
fi
"""
                post_cmd_lines.append(cmd)

            write_bash_script(post_script_path, "\n".join(post_cmd_lines))

            # 3. Add Rules
            gen_sentinel = f"{LOCAL_JOBS_DIR}/gen/gen_{chunk_id}.done"
            post_sentinel = f"{LOCAL_JOBS_DIR}/post/post_{chunk_id}.done"
            
            sample_post_sentinels.append(post_sentinel)
            
            rules_content += f"""
rule gen_{chunk_id}:
    output:
        \"{gen_sentinel}\" 
    shell:
        \"{gen_script_path} && touch {{output}}\" 

rule post_{chunk_id}:
    input:
        \"{gen_sentinel}\" 
    output:
        \"{post_sentinel}\" 
    shell:
        \"{post_script_path} && touch {{output}}\" 
"""
        # Collect all postprocessing sentinels for this sample to act as deps for TFDS
        all_sample_post_sentinels[sample_key] = sample_post_sentinels


    # -------------------------------------------------------------------------
    # PART 2: TFDS Conversion (Per Config ID / Split)
    # -------------------------------------------------------------------------
    tfds_sentinels = []
    
    for sample_key, mapping in tfds_mappings.items():
        if sample_key not in samples:
            print(f"Warning: TFDS mapping found for {sample_key} but no sample definition.")
            continue
            
        sample_data = samples[sample_key]
        builder_path = mapping['builder_path']
        config_ids = mapping['config_ids']
        
        manual_dir = os.path.join(workspace_dir, sample_data['output_subdir'])
        
        for config_id in config_ids:
            tfds_id = f"{sample_key}_tfds_{config_id}"
            tfds_script_path = f"{LOCAL_JOBS_DIR}/tfds/tfds_{tfds_id}.sh"
            tfds_sentinel = f"{LOCAL_JOBS_DIR}/tfds/tfds_{tfds_id}.done"
            
            tfds_build_cmd = f"tfds build {builder_path} --config {config_id} --data_dir {tfds_root_dir} --manual_dir {manual_dir} --overwrite"
            if container_img:
                tfds_build_cmd = f"apptainer exec {bind_args} {container_img} {tfds_build_cmd}"

            cmd = f"""
export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR={tfds_root_dir}

echo "Building TFDS for {builder_path} config {config_id}"
echo "Manual dir: {manual_dir}"

{tfds_build_cmd}
"""
            write_bash_script(tfds_script_path, cmd)
            
            input_sentinels_str = ',\n        '.join([f'\"{s}\"' for s in all_sample_post_sentinels[sample_key]])
            
            rules_content += f"""
rule tfds_{tfds_id}:
    input:
        {input_sentinels_str}
    output:
        \"{tfds_sentinel}\" 
    shell:
        \"{tfds_script_path} && touch {{output}}\" 
"""
            tfds_sentinels.append(tfds_sentinel)

    # -------------------------------------------------------------------------
    # Finalize Snakefile
    # -------------------------------------------------------------------------

    def fmt_list(lst):
        return "[" + ", ".join([f'\"{x}\"' for x in lst]) + "]"

    snakefile_content += "        " + fmt_list(tfds_sentinels) + "\n"
    snakefile_content += rules_content

    with open(f"{LOCAL_JOBS_DIR}/Snakefile", "w") as f:
        f.write(snakefile_content)
    
    print(f"Generated Snakemake workflow in {LOCAL_JOBS_DIR}/Snakefile")
    print(f"Generated {len(tfds_sentinels)} TFDS jobs.")

if __name__ == "__main__":
    main()
