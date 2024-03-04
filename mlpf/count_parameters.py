import sys
import yaml

sys.path.append("../mlpf")

from pyg.mlpf import MLPF
from pyg.utils import (
    CLASS_LABELS,
    X_FEATURES,
    count_parameters,
)


with open(sys.argv[1], "r") as stream:  # load config (includes: which physics samples, model params)
    config = yaml.safe_load(stream)

model_kwargs = {
    "input_dim": len(X_FEATURES[config["dataset"]]),
    "num_classes": len(CLASS_LABELS[config["dataset"]]),
    "pt_mode": config["model"]["pt_mode"],
    "eta_mode": config["model"]["eta_mode"],
    "sin_phi_mode": config["model"]["sin_phi_mode"],
    "cos_phi_mode": config["model"]["cos_phi_mode"],
    "energy_mode": config["model"]["energy_mode"],
    "attention_type": config["model"]["attention"]["attention_type"],
    **config["model"][config["conv_type"]],
}
model = MLPF(**model_kwargs)

trainable_params, nontrainable_params, table = count_parameters(model)

print(table)

print("Model conv type:", model.conv_type)
print("conv_type HPs", config["model"][config["conv_type"]])
print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", nontrainable_params)
print("Total parameters:", trainable_params + nontrainable_params)
