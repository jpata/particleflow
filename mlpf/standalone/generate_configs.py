import random
import json
import os
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


def generate_random_config():
    # Common parameters
    emb_dim = random.choice([128, 256])
    n_heads = random.choice([8, 16]) if emb_dim == 128 else random.choice([16, 32])
    # Ensure divisibility
    if emb_dim % n_heads != 0:
        n_heads = 16 if emb_dim == 128 else 32

    width = emb_dim * 4

    # 1. Input
    i_type = random.choice(["default", "projection_only"])
    input_str = f"i(55,{emb_dim},{emb_dim*2},{i_type})"

    # 2. Layers helpers
    def get_layer_expr(ltype=None, n=None):
        t = ltype or random.choice(["h", "g", "s", "f"])
        num = n or random.randint(1, 4)
        p = random.choice(["T", "F"])
        return f"{t}({n_heads},{emb_dim},{width},pos={p})*{num}"

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
    output_str = f"o(8,{emb_dim*2},default,rg={rg})"

    return f"{input_str}|{backbone_str}|{output_str}"


def mutate_layer(layer: LayerConfig) -> LayerConfig:
    # Randomly mutate layer parameters
    params = {
        "num_heads": layer.num_heads,
        "embedding_dim": layer.embedding_dim,
        "width": layer.width,
        "pos": layer.pos,
    }

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

    # Ensure divisibility
    if params["embedding_dim"] % params["num_heads"] != 0:
        params["num_heads"] = 16 if params["embedding_dim"] == 128 else 32

    # Re-create the appropriate layer type
    if isinstance(layer, HEPTConfig):
        return HEPTConfig(**params)
    elif isinstance(layer, GlobalConfig):
        return GlobalConfig(**params)
    elif isinstance(layer, StandardConfig):
        return StandardConfig(**params)
    elif isinstance(layer, FastformerConfig):
        return FastformerConfig(**params)
    return LayerConfig(type=layer.type, **params)


def mutate_config(config: ModelConfig) -> ModelConfig:
    # Randomly mutate part of the configuration
    mutation_type = random.choice(["input", "backbone_layer", "backbone_structure", "output"])

    new_input = config.input
    new_backbone = {k: list(v) for k, v in config.backbone.items()}
    new_output = config.output

    if mutation_type == "input":
        i_type = random.choice(["default", "projection_only"])
        new_input = InputConfig(config.input.input_dim, config.input.embedding_dim, config.input.width, i_type)

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
                new_layer = parse_dsl(generate_random_config()).backbone["shared"][0]
            new_backbone[branch].append(new_layer)
        elif len(new_backbone[branch]) > 1:
            # Remove layer
            new_backbone[branch].pop(random.randint(0, len(new_backbone[branch]) - 1))

    elif mutation_type == "output":
        rg = random.choice(["direct", "additive", "linear", "{pt:linear,eta:additive}"])
        # We need to parse rg if it's a string representation of a dict or use it directly
        if "{" in rg:
            rg_val = {"pt": "linear", "eta": "additive"}
        else:
            rg_val = rg
        new_output = OutputConfig(config.output.num_classes, config.output.width, config.output.type, rg_val)

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
            if random.random() < 0.8:
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


if __name__ == "__main__":
    # Example usage:
    # If population_metrics.json exists, evolve from it.
    # Otherwise, generate random population.

    POP_SIZE = 100
    metrics_file = "population_metrics.json"

    if os.path.exists(metrics_file):
        print(f"Loading existing population from {metrics_file}...")
        with open(metrics_file, "r") as f:
            data = json.load(f)

        pop_with_fitness = []
        for dsl, metrics in data.items():
            try:
                cfg = parse_dsl(dsl)
                # Define fitness:
                # lower val_jet_iqr is better
                # higher val_jet_matched_frac is better
                # lower runtime_cpu_ms is better
                iqr = metrics.get("val_jet_iqr", 2.0)
                matched_frac = metrics.get("val_jet_matched_frac", 0.5)
                runtime_cpu = metrics.get("runtime_cpu_ms", 1000.0)

                # Higher is better
                fitness = matched_frac / (iqr * (1.0 + runtime_cpu / 1000.0))
                pop_with_fitness.append((cfg, fitness))
            except Exception:
                continue

        if len(pop_with_fitness) > 10:
            print(f"Evolving from {len(pop_with_fitness)} configurations...")
            new_pop_configs = evolve(pop_with_fitness, pop_size=POP_SIZE)
        else:
            print("Not enough successful configurations to evolve. Generating random...")
            new_pop_configs = [parse_dsl(generate_random_config()) for _ in range(POP_SIZE)]
    else:
        print("No existing metrics found. Generating random population...")
        new_pop_configs = []
        while len(new_pop_configs) < POP_SIZE:
            try:
                dsl = generate_random_config()
                cfg = parse_dsl(dsl)
                new_pop_configs.append(cfg)
            except Exception:
                continue

    # Save to configs.txt
    configs = [config_to_string(cfg) for cfg in new_pop_configs]
    with open("configs.txt", "w") as f:
        for c in configs:
            f.write(c + "\n")

    print(f"\nSuccessfully generated and validated {len(configs)} configurations.")
    print("Saved to configs.txt")
