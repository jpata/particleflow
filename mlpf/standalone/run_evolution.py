import argparse
import json
import os
import re
import random
import glob
import subprocess
from queue import Queue
from threading import Thread

from mlpf.standalone.dsl import (
    parse_dsl,
    config_to_string,
    ModelConfig,
    InputConfig,
    OutputConfig,
    LayerConfig,
    HEPTConfig,
    GlobalConfig,
    StandardConfig,
    FastformerConfig,
)


def get_available_gpus():
    all_uuids = []
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        for i, line in enumerate(lines):
            # If the current line is a GPU, check if the next line is a MIG device
            # If it is, we skip the parent GPU and only use the MIG devices
            if line.startswith("GPU"):
                if i + 1 < len(lines) and "MIG" in lines[i + 1]:
                    continue

            # Extract UUID
            match = re.search(r"UUID: ([\w-]+)", line)
            if match:
                all_uuids.append(match.group(1))
    except Exception:
        pass

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        cvd = [g.strip() for g in cvd if g.strip()]

        # If they are already UUIDs, return as is
        if cvd and (cvd[0].startswith("GPU-") or cvd[0].startswith("MIG-")):
            return cvd

        # If they are indices, try to map to discovered UUIDs
        if all_uuids:
            mapped_gpus = []
            for g in cvd:
                try:
                    idx = int(g)
                    if 0 <= idx < len(all_uuids):
                        mapped_gpus.append(all_uuids[idx])
                    else:
                        mapped_gpus.append(g)
                except ValueError:
                    mapped_gpus.append(g)
            return mapped_gpus
        return cvd

    if all_uuids:
        return all_uuids

    return ["0"]


def generate_random_config():
    # Common parameters
    emb_dim = random.choice([128, 256])
    n_heads = random.choice([8, 16]) if emb_dim == 128 else random.choice([16, 32])
    # Ensure divisibility
    if emb_dim % n_heads != 0:
        n_heads = 16 if emb_dim == 128 else 32

    width = emb_dim * 4
    dropout = random.choice([0.0, 0.1, 0.2])

    # 1. Input
    i_type = random.choice(["default", "projection_only"])
    input_str = f"i(55,{emb_dim},{emb_dim*2},{i_type},dropout={dropout})"

    # 2. Layers helpers
    def get_layer_expr(ltype=None, n=None):
        t = ltype or random.choice(["h", "g", "s", "f"])
        num = n or random.randint(1, 4)
        p = random.choice(["T", "F"])
        d = random.choice([0.0, 0.1, 0.2])

        extra_params = ""
        if t == "h" and random.random() < 0.4:
            if random.random() < 0.5:
                extra_params += f",block_size={random.choice([50, 100, 200, 400])}"
            if random.random() < 0.5:
                extra_params += f",n_hashes={random.choice([2, 3, 4, 5])}"

        return f"{t}({n_heads},{emb_dim},{width},pos={p},dropout={d}{extra_params})*{num}"

    # 3. Backbone structure
    b_type = random.choice(["joint", "split", "hybrid"])
    if b_type == "joint":
        backbone_str = get_layer_expr(n=random.randint(2, 8))
    elif b_type == "split":
        backbone_str = "{" + f"pid:{get_layer_expr()},reg:{get_layer_expr()}" + "}"
    else:  # hybrid
        backbone_str = f"{get_layer_expr(n=random.randint(1, 3))}+" + "{" + f"pid:{get_layer_expr()},reg:{get_layer_expr()}" + "}"

    # 4. Output
    rg = random.choice(["direct", "additive", "linear", "{pt:linear,eta:additive}"])
    d_out = random.choice([0.0, 0.1, 0.2])
    output_str = f"o(8,{emb_dim*2},default,rg={rg},dropout={d_out})"

    return f"{input_str}|{backbone_str}|{output_str}"


def mutate_layer(layer: LayerConfig) -> LayerConfig:
    # Randomly mutate layer parameters
    params = {
        "num_heads": layer.num_heads,
        "embedding_dim": layer.embedding_dim,
        "width": layer.width,
        "pos": layer.pos,
        "dropout": layer.dropout,
    }

    hept_fields = []
    if isinstance(layer, HEPTConfig):
        hept_fields = ["block_size", "n_hashes", "num_regions", "num_w_per_dist"]
        for field in hept_fields:
            params[field] = getattr(layer, field)

    # Mutate one parameter
    p = random.choice(list(params.keys()))
    if p == "num_heads":
        params[p] = random.choice([8, 16, 32])
    elif p == "embedding_dim":
        params[p] = random.choice([128, 256])
    elif p == "width":
        params[p] = params["embedding_dim"] * random.choice([2, 4, 8])
    elif p == "pos":
        params[p] = not params[p]
    elif p == "dropout":
        params[p] = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])
    elif p == "block_size":
        params[p] = random.choice([50, 100, 200, 400])
    elif p == "n_hashes":
        params[p] = random.choice([2, 3, 4, 5, 6])
    elif p == "num_regions":
        params[p] = random.randint(100, 300)
    elif p == "num_w_per_dist":
        params[p] = random.randint(5, 20)

    # Ensure divisibility
    if params["embedding_dim"] % params["num_heads"] != 0:
        params["num_heads"] = 16 if params["embedding_dim"] == 128 else 32

    # Re-create the appropriate layer type
    if isinstance(layer, HEPTConfig):
        return HEPTConfig(**params)

    # Filter only base parameters for non-HEPT layers
    base_params = {k: params[k] for k in ["num_heads", "embedding_dim", "width", "pos", "dropout"]}
    if isinstance(layer, GlobalConfig):
        return GlobalConfig(**base_params)
    elif isinstance(layer, StandardConfig):
        return StandardConfig(**base_params)
    elif isinstance(layer, FastformerConfig):
        return FastformerConfig(**base_params)
    return LayerConfig(type=layer.type, **base_params)


def mutate_config(config: ModelConfig) -> ModelConfig:
    # Randomly mutate part of the configuration
    mutation_type = random.choice(["input", "backbone_layer", "backbone_structure", "output"])

    new_input = config.input
    new_backbone = {k: list(v) for k, v in config.backbone.items()}
    new_output = config.output

    if mutation_type == "input":
        i_type = random.choice(["default", "projection_only"])
        dropout = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])
        new_input = InputConfig(config.input.input_dim, config.input.embedding_dim, config.input.width, i_type, dropout)

    elif mutation_type == "backbone_layer":
        # Mutate a random layer in a random branch
        branch = random.choice([k for k, v in new_backbone.items() if v])
        idx = random.randint(0, len(new_backbone[branch]) - 1)
        new_backbone[branch][idx] = mutate_layer(new_backbone[branch][idx])

    elif mutation_type == "backbone_structure":
        # Add or remove a layer
        branch = random.choice(list(new_backbone.keys()))
        if random.random() < 0.5 and len(new_backbone[branch]) < 10:
            # Add layer (copy from existing or create new)
            if new_backbone[branch]:
                new_layer = random.choice(new_backbone[branch])
            else:
                rand_cfg = parse_dsl(generate_random_config())
                all_layers = [layer for branch_layers in rand_cfg.backbone.values() for layer in branch_layers]
                new_layer = random.choice(all_layers)
            new_backbone[branch].append(new_layer)
        elif len(new_backbone[branch]) > 1:
            # Remove layer
            new_backbone[branch].pop(random.randint(0, len(new_backbone[branch]) - 1))

    elif mutation_type == "output":
        rg_modes = ["direct", "additive", "linear", "multiplicative"]

        if random.random() < 0.3:
            # Mutate one of the individual target modes
            if isinstance(config.output.rg_mode, dict):
                rg_val = dict(config.output.rg_mode)
            else:
                rg_val = {k: config.output.rg_mode for k in ["pt", "eta", "sin_phi", "cos_phi", "energy"]}

            target = random.choice(list(rg_val.keys()))
            rg_val[target] = random.choice(rg_modes)
        else:
            # Global rg mode change
            rg_val = random.choice(rg_modes)

        dropout = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])
        emb_dim = config.output.embedding_dim
        if random.random() < 0.2:
            emb_dim = random.choice([None, 128, 256])

        new_output = OutputConfig(config.output.num_classes, config.output.width, config.output.type, rg_val, dropout, emb_dim)

    return ModelConfig(new_input, new_backbone, new_output)


def crossover(parent1: ModelConfig, parent2: ModelConfig) -> ModelConfig:
    # Swap components between parents
    new_input = random.choice([parent1.input, parent2.input])
    new_output = random.choice([parent1.output, parent2.output])

    # For backbone, we can swap branches
    new_backbone = {}
    for k in set(parent1.backbone.keys()) | set(parent2.backbone.keys()):
        new_backbone[k] = list(random.choice([parent1.backbone.get(k, []), parent2.backbone.get(k, [])]))

    return ModelConfig(new_input, new_backbone, new_output)


def evolve(population_with_fitness, pop_size=100, mutation_rate=0.2):
    # population_with_fitness is a list of (ModelConfig, fitness)
    # Sort by fitness descending
    population_with_fitness.sort(key=lambda x: x[1], reverse=True)

    new_population = []
    # Elitism: keep top 10%
    elitism_count = max(1, pop_size // 10)
    for i in range(elitism_count):
        new_population.append(population_with_fitness[i][0])

    # Fill the rest
    while len(new_population) < pop_size:
        if random.random() < 0.5:
            # Crossover
            p1 = random.choice(population_with_fitness[: pop_size // 2])[0]
            p2 = random.choice(population_with_fitness[: pop_size // 2])[0]
            child = crossover(p1, p2)
        else:
            # Mutation or random
            if random.random() < (1.0 - mutation_rate):
                parent = random.choice(population_with_fitness[: pop_size // 2])[0]
                child = mutate_config(parent)
            else:
                child = parse_dsl(generate_random_config())

        try:
            child.validate()
            new_population.append(child)
        except Exception:
            continue

    return new_population


def parse_log(file_path):
    if not os.path.exists(file_path):
        return None, None

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
        "val_loss": r"val_loss\s+:\s+([\d\.]+)",
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


def worker_fn(gpu_id, task_queue, log_dir, data_dir, apptainer_cmd, verbose):
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, dsl = task

        log_file = os.path.join(log_dir, f"job_{idx}.out")
        err_file = os.path.join(log_dir, f"job_{idx}.err")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Determine paths relative to this script or execution dir
        eval_script = "mlpf/standalone/eval.py"
        if not os.path.exists(eval_script):
            eval_script = os.path.join(os.path.dirname(__file__), "eval.py")

        cmd = ["python3", eval_script, "--data-dir", data_dir, "--dsl", dsl]

        if apptainer_cmd:
            cmd = apptainer_cmd.split() + cmd

        if verbose:
            print(f"[{gpu_id}] Running task {idx} with DSL: {dsl}")

        with open(log_file, "w") as fout, open(err_file, "w") as ferr:
            fout.write(f"Using DSL: {dsl}\n")
            fout.flush()
            subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)

        if verbose:
            print(f"[{gpu_id}] Finished task {idx}")

        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Run full genetic algorithm optimization chain on a multi-GPU node.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to tfds directory")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations to evolve")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size per generation")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated list of GPUs to use (defaults to all available)")
    parser.add_argument("--log-dir", type=str, default="logs/evolution", help="Directory for logs")
    parser.add_argument("--apptainer", type=str, default="", help="Optional apptainer command prefix to wrap eval.py")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose worker logs")
    args = parser.parse_args()

    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",")]
    else:
        gpus = get_available_gpus()

    print(f"Using {len(gpus)} GPUs: {gpus}")

    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize first generation
    print(f"Generating initial random population of size {args.pop_size}...")
    pop_configs = []
    while len(pop_configs) < args.pop_size:
        try:
            dsl = generate_random_config()
            cfg = parse_dsl(dsl)
            pop_configs.append(cfg)
        except Exception:
            continue

    for gen in range(1, args.generations + 1):
        print("\n" + "=" * 40)
        print(f"       GENERATION {gen} / {args.generations}")
        print("========================================")

        gen_log_dir = os.path.join(args.log_dir, f"gen_{gen}")
        os.makedirs(gen_log_dir, exist_ok=True)

        task_queue = Queue()

        for idx, cfg in enumerate(pop_configs):
            task_queue.put((idx, config_to_string(cfg)))

        threads = []
        for gpu_id in gpus:
            t = Thread(target=worker_fn, args=(gpu_id, task_queue, gen_log_dir, args.data_dir, args.apptainer, args.verbose))
            t.start()
            threads.append(t)

        # Wait for all tasks to finish
        print(f"Launched {args.pop_size} evaluations across {len(gpus)} GPUs. Waiting for completion...")
        task_queue.join()

        # Stop workers
        for _ in gpus:
            task_queue.put(None)
        for t in threads:
            t.join()

        # Collect metrics
        print(f"\nEvaluating generation {gen} results...")
        log_files = glob.glob(os.path.join(gen_log_dir, "job_*.out"))

        pop_with_fitness = []
        gen_metrics = {}

        for log_file in log_files:
            try:
                dsl, metrics = parse_log(log_file)
                if dsl and metrics:
                    gen_metrics[dsl] = metrics

                    iqr = metrics.get("val_jet_iqr", 2.0)
                    matched_frac = metrics.get("val_jet_matched_frac", 0.5)
                    runtime_cpu = metrics.get("runtime_cpu_ms", 1000.0)
                    val_loss = metrics.get("val_loss", 10.0)

                    # Individual terms for fitness
                    term_matching = matched_frac
                    term_iqr = 1.0 / max(iqr, 0.01)
                    term_loss = 1.0 / (1.0 + val_loss)
                    term_runtime = 1.0 / (1.0 + runtime_cpu / 1000.0)

                    # Total fitness
                    print(f"fitness matching={term_matching:.2f} iqr={term_iqr:.2f} loss={term_loss:.2f} runtime={term_runtime:.2f}")
                    fitness = term_matching * term_iqr * term_loss * term_runtime

                    cfg = parse_dsl(dsl)
                    pop_with_fitness.append((cfg, fitness))
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")

        # Save metrics for this generation
        metrics_file = os.path.join(args.log_dir, f"gen_{gen}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(gen_metrics, f, indent=4)

        print(f"Valid evaluations: {len(pop_with_fitness)} / {args.pop_size}")

        if pop_with_fitness:
            pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
            print(f"\nTop 3 DSLs for generation {gen}:")
            top_3 = pop_with_fitness[:3]
            for i, (cfg, fitness) in enumerate(top_3):
                print(f"  {i+1}. Fitness: {fitness:.6f} | DSL: {config_to_string(cfg)}")

            mean_top_3 = sum(f for c, f in top_3) / len(top_3)
            print(f"Mean top-3 fitness: {mean_top_3:.6f}")

        if gen < args.generations:
            if len(pop_with_fitness) > max(5, args.pop_size // 10):
                print("Evolving next generation...")
                pop_configs = evolve(pop_with_fitness, pop_size=args.pop_size)
            else:
                print("Not enough successful configurations to evolve. Generating random...")
                pop_configs = []
                while len(pop_configs) < args.pop_size:
                    try:
                        cfg = parse_dsl(generate_random_config())
                        pop_configs.append(cfg)
                    except Exception:
                        continue

        print(f"Generation {gen} complete.")


if __name__ == "__main__":
    main()
