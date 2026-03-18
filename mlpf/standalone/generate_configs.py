import random
from mlpf.standalone.dsl import parse_dsl, config_to_string


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


configs = []
valid_count = 0

print("Generating and validating 100 configurations in the new format...")

for i in range(100):
    try:
        dsl = generate_random_config()
        # Validate by parsing
        cfg = parse_dsl(dsl)
        # Normalize by stringifying back (this will produce the new compact format)
        normalized_dsl = config_to_string(cfg)

        # Double check parsing of the normalized version
        parse_dsl(normalized_dsl)

        configs.append(normalized_dsl)
        valid_count += 1
    except Exception:
        # print(f"Generation error: {e}")
        continue

# Save to file
with open("configs.txt", "w") as f:
    for c in configs:
        f.write(c + "\n")

print(f"\nSuccessfully generated and validated {len(configs)} configurations.")
print("Saved to configs.txt")

# Final check of first 5
print("\nSample configurations:")
for c in configs[:5]:
    print(c)
