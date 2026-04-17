from dataclasses import dataclass
from typing import List, Dict, Union, Optional


class LayerConfig:
    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        if not isinstance(other, LayerConfig):
            return False
        return self.__dict__ == other.__dict__


@dataclass(frozen=True)
class HEPTConfig(LayerConfig):
    num_heads: int = 16
    embedding_dim: int = 128
    width: int = 512
    pos: bool = False
    dropout: float = 0.1
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
    input_dim: int = 55
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


def i(input_dim=55, embedding_dim=128, width=256, type="default", dropout=0.1):
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
    Example: 'i(55,128,256,default)|h(16,128,512,pos=T)*6|o(8,256,default,rg=linear)'
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
    }

    def eval_part(p):
        if "*" in p:
            base, count = p.split("*")
            return [eval(base, {"__builtins__": None}, ns)] * int(count)
        else:
            return [eval(p, {"__builtins__": None}, ns)]

    try:
        input_cfg = eval_part(parts[0])[0]
        backbone_raw = parts[1].split("+")
        backbone = {}

        for b in backbone_raw:
            if ":" in b:
                key, layers_str = b.split(":")
                backbone[key] = []
                for _l in layers_str.split(","):
                    backbone[key].extend(eval_part(_l))
            else:
                if "shared" not in backbone:
                    backbone["shared"] = []
                backbone["shared"].extend(eval_part(b))

        output_cfg = eval_part(parts[2])[0]

        return ModelConfig(input=input_cfg, backbone=backbone, output=output_cfg)
    except Exception as e:
        raise ValueError(f"Failed to parse DSL: {e}")


def config_to_string(config: ModelConfig) -> str:
    """Serializes a ModelConfig back to a DSL string."""

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
            func = type_to_func.get(curr.type, curr.type)
            params = f"{curr.num_heads},{curr.embedding_dim},{curr.width}"
            if curr.pos:
                params += ",pos=T"
            if hasattr(curr, "dropout") and curr.dropout != 0.1:
                params += f",dropout={curr.dropout}"

            # Extra params for HEPT
            if curr.type == "hept":
                for _field in ["block_size", "n_hashes", "num_regions", "num_w_per_dist"]:
                    val = getattr(curr, _field)
                    # only add if different from default
                    if val != HEPTConfig.__dataclass_fields__[_field].default:
                        params += f",{_field}={val}"

            s = f"{func}({params})"
            if count > 1:
                s += f"*{count}"
            res.append(s)
            i += count
        return ",".join(res)

    input_str = f"i({config.input.input_dim},{config.input.embedding_dim},{config.input.width},{config.input.type},{config.input.dropout})"

    backbone_parts = []
    if "shared" in config.backbone:
        backbone_parts.append(layers_to_str(config.backbone["shared"]))

    for k in ["pid", "reg", "binary"]:
        if k in config.backbone and config.backbone[k]:
            backbone_parts.append(f"{k}:{layers_to_str(config.backbone[k])}")

    backbone_str = "+".join(backbone_parts)

    rg_str = config.output.rg_mode
    if isinstance(rg_str, dict):
        rg_str = "{" + ",".join([f"{k}={v}" for k, v in rg_str.items()]) + "}"

    output_str = f"o({config.output.num_classes},{config.output.width},{config.output.type},{rg_str},{config.output.dropout})"
    if config.output.embedding_dim:
        output_str = output_str[:-1] + f",embedding_dim={config.output.embedding_dim})"

    return f"{input_str}|{backbone_str}|{output_str}"
