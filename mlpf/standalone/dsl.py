from dataclasses import dataclass
from typing import List, Union, Dict, Optional


@dataclass(frozen=True)
class LayerConfig:
    type: str
    num_heads: int = 16
    embedding_dim: int = 128
    width: int = 512
    dropout: float = 0.1
    pos: bool = False

    def __mul__(self, n: int):
        return [self for _ in range(n)]

    def __add__(self, other):
        if isinstance(other, list):
            return [self] + other
        return [self, other]

    def __radd__(self, other):
        if isinstance(other, list):
            return other + [self]
        return [other, self]


@dataclass(frozen=True)
class HEPTConfig(LayerConfig):
    type: str = "hept"
    block_size: int = 100
    n_hashes: int = 3
    num_regions: int = 140
    num_w_per_dist: int = 10


@dataclass(frozen=True)
class GlobalConfig(LayerConfig):
    type: str = "global"


@dataclass(frozen=True)
class StandardConfig(LayerConfig):
    type: str = "standard"


@dataclass(frozen=True)
class FastformerConfig(LayerConfig):
    type: str = "fastformer"


@dataclass(frozen=True)
class InputConfig:
    input_dim: int = 73
    embedding_dim: int = 128
    width: int = 256
    type: str = "default"  # "default" or "projection_only" or "advanced"
    dropout: float = 0.1


@dataclass(frozen=True)
class OutputConfig:
    num_classes: int = 8
    width: int = 256
    type: str = "default"
    rg_mode: Union[str, Dict[str, str]] = "direct"
    dropout: float = 0.1
    embedding_dim: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    input: InputConfig
    backbone: Dict[str, List[LayerConfig]]
    output: OutputConfig

    def validate(self):
        # Type checking without running the code
        if self.input.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        def check_backbone(layers, prev_dim):
            d = prev_dim
            for layer in layers:
                if layer.embedding_dim % layer.num_heads != 0:
                    raise ValueError(f"embedding_dim ({layer.embedding_dim}) must be divisible by num_heads ({layer.num_heads}) for {layer.type}")
                if layer.embedding_dim != d:
                    raise ValueError(
                        f"Dimension mismatch: layer {layer.type} expects embedding_dim={layer.embedding_dim} but previous layer/input produced {d}"
                    )
                d = layer.embedding_dim
            return d

        shared_dim = check_backbone(self.backbone.get("shared", []), self.input.embedding_dim)

        for k in ["pid", "reg", "binary"]:
            if self.backbone.get(k):
                check_backbone(self.backbone[k], shared_dim)

        print("ModelConfig validated successfully.")


def i(input_dim=73, embedding_dim=128, width=256, type="default", dropout=0.1):
    return InputConfig(input_dim, embedding_dim, width, str(type), float(dropout))


def h(num_heads=16, embedding_dim=128, width=512, pos=False, dropout=0.1, **kwargs):
    return HEPTConfig(num_heads=num_heads, embedding_dim=embedding_dim, width=width, pos=bool(pos), dropout=float(dropout), **kwargs)


def g(num_heads=16, embedding_dim=128, width=512, pos=False, dropout=0.1):
    return GlobalConfig(num_heads=num_heads, embedding_dim=embedding_dim, width=width, pos=bool(pos), dropout=float(dropout))


def s(num_heads=16, embedding_dim=128, width=512, pos=False, dropout=0.1):
    return StandardConfig(num_heads=num_heads, embedding_dim=embedding_dim, width=width, pos=bool(pos), dropout=float(dropout))


def f(num_heads=16, embedding_dim=128, width=512, pos=False, dropout=0.1):
    return FastformerConfig(num_heads=num_heads, embedding_dim=embedding_dim, width=width, pos=bool(pos), dropout=float(dropout))


def o(num_classes=8, width=256, type="default", rg="direct", dropout=0.1, embedding_dim=None):
    return OutputConfig(num_classes, width, str(type), rg, float(dropout), embedding_dim)


def parse_dsl(dsl_str: str) -> ModelConfig:
    """
    Parses a DSL string without spaces or quotes.
    Example: 'i(73,128,256,default)|h(16,128,512,pos=T)*6|o(8,256,default,rg=linear)'
    """
    dsl_str = dsl_str.replace(" ", "")
    parts = dsl_str.split("|")
    if len(parts) != 3:
        raise ValueError("DSL must have 3 parts separated by '|': input|backbone|output")

    # Namespace for evaluation - allow identifiers to be treated as strings
    ns = {
        "i": i,
        "h": h,
        "g": g,
        "s": s,
        "f": f,
        "o": o,
        "input": i,
        "hept": h,
        "global": g,
        "standard": s,
        "fastformer": f,
        "output": o,
        "True": True,
        "False": False,
        "T": True,
        "F": False,
        "default": "default",
        "projection_only": "projection_only",
        "direct": "direct",
        "additive": "additive",
        "multiplicative": "multiplicative",
        "linear": "linear",
        "shared": "shared",
        "pid": "pid",
        "reg": "reg",
        "binary": "binary",
        "pt": "pt",
        "eta": "eta",
        "sin_phi": "sin_phi",
        "cos_phi": "cos_phi",
        "energy": "energy",
    }

    # 1. Parse input
    input_cfg = eval(parts[0], {"__builtins__": {}}, ns)

    # 2. Parse backbone
    backbone_str = parts[1].strip()
    final_backbone = {"shared": [], "pid": [], "reg": [], "binary": []}

    if "{" in backbone_str:
        if backbone_str.count("{") == 1 and backbone_str.find("{") > 0:
            idx = backbone_str.find("{")
            joint_expr = backbone_str[:idx].strip()
            if joint_expr.endswith("+"):
                joint_expr = joint_expr[:-1]
            if joint_expr:
                joint_layers = eval(joint_expr, {"__builtins__": {}}, ns)
                final_backbone["shared"] = joint_layers if isinstance(joint_layers, list) else [joint_layers]
            dict_str = backbone_str[idx:]
        else:
            dict_str = backbone_str

        dict_val = eval(dict_str, {"__builtins__": {}}, ns)
        for k, v in dict_val.items():
            layers = v if isinstance(v, list) else [v]
            if k == "shared":
                final_backbone["shared"].extend(layers)
            else:
                final_backbone[k] = layers
    else:
        layers = eval(backbone_str, {"__builtins__": {}}, ns)
        final_backbone["shared"] = layers if isinstance(layers, list) else [layers]

    # 3. Parse output
    output_cfg = eval(parts[2], {"__builtins__": {}}, ns)

    config = ModelConfig(input_cfg, final_backbone, output_cfg)
    config.validate()
    return config


def config_to_string(cfg: ModelConfig) -> str:
    def layers_to_str(layers):
        if not layers:
            return ""
        res = []
        i = 0
        type_to_func = {"hept": "h", "global": "g", "standard": "s", "fastformer": "f"}
        while i < len(layers):
            curr = layers[i]
            count = 1
            while i + 1 < len(layers) and layers[i + 1] == curr:
                count += 1
                i += 1
            p_str = ",pos=T" if curr.pos else ""
            d_str = f",dropout={curr.dropout}" if curr.dropout != 0.1 else ""
            params = f"{curr.num_heads},{curr.embedding_dim},{curr.width}{p_str}{d_str}"
            if isinstance(curr, HEPTConfig):
                if curr.block_size != 100:
                    params += f",block_size={curr.block_size}"
                if curr.n_hashes != 3:
                    params += f",n_hashes={curr.n_hashes}"
                if curr.num_regions != 140:
                    params += f",num_regions={curr.num_regions}"
                if curr.num_w_per_dist != 10:
                    params += f",num_w_per_dist={curr.num_w_per_dist}"

            func = type_to_func.get(curr.type, curr.type[0])
            layer_str = f"{func}({params})"
            if count > 1:
                layer_str += f"*{count}"
            res.append(layer_str)
            i += 1
        return "+".join(res)

    d_str = f",dropout={cfg.input.dropout}" if cfg.input.dropout != 0.1 else ""
    i_str = f"i({cfg.input.input_dim},{cfg.input.embedding_dim},{cfg.input.width},{cfg.input.type}{d_str})"
    shared_str = layers_to_str(cfg.backbone.get("shared", []))
    branches = {k: v for k, v in cfg.backbone.items() if k != "shared" and v}

    if branches:
        branch_str = "{" + ",".join([f"{k}:{layers_to_str(v)}" for k, v in branches.items()]) + "}"
        b_str = f"{shared_str}+{branch_str}" if shared_str else branch_str
    else:
        b_str = shared_str

    rg_val = cfg.output.rg_mode
    if isinstance(rg_val, dict):
        rg_str = "{" + ",".join([f"{k}:{v}" for k, v in rg_val.items()]) + "}"
    else:
        rg_str = str(rg_val)

    d_str = f",dropout={cfg.output.dropout}" if cfg.output.dropout != 0.1 else ""
    e_str = f",embedding_dim={cfg.output.embedding_dim}" if cfg.output.embedding_dim is not None else ""
    o_str = (
        f"o({cfg.output.num_classes},{cfg.output.width},{cfg.output.type},rg={rg_str}{d_str}{e_str})"
        if rg_val != "direct"
        else f"o({cfg.output.num_classes},{cfg.output.width},{cfg.output.type}{d_str}{e_str})"
    )
    return f"{i_str}|{b_str}|{o_str}"
