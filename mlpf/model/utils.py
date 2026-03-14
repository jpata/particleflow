import json
import pickle as pkl

import pandas as pd
import torch
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ConstantLR
import logging
from mlpf.conf import (
    MLPFConfig,
    Y_FEATURES,
    LRSchedule,
)

# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254


def unpack_target(y, model):
    ret = {}
    ret["cls_id"] = y[..., 0].long()
    ret["charge"] = torch.clamp((y[..., 1] + 1).to(dtype=torch.float32), 0, 2)  # -1, 0, 1 -> 0, 1, 2

    for i, feat in enumerate(Y_FEATURES):
        if i >= 2:  # skip the cls and charge as they are defined above
            ret[feat] = y[..., i].to(dtype=torch.float32)
    ret["phi"] = torch.atan2(ret["sin_phi"], ret["cos_phi"])
    ret["ispu"] = (ret["ispu"] == 1).to(dtype=torch.float32)
    # do some sanity checks
    # assert torch.all(ret["pt"] >= 0.0)  # pt
    # assert torch.all(torch.abs(ret["sin_phi"]) <= 1.0)  # sin_phi
    # assert torch.all(torch.abs(ret["cos_phi"]) <= 1.0)  # cos_phi
    # assert torch.all(ret["energy"] >= 0.0)  # energy

    # note ~ momentum = ["pt", "eta", "sin_phi", "cos_phi", "energy"]
    ret["momentum"] = y[..., 2:7].to(dtype=torch.float32)
    ret["p4"] = torch.cat([ret["pt"].unsqueeze(-1), ret["eta"].unsqueeze(-1), ret["phi"].unsqueeze(-1), ret["energy"].unsqueeze(-1)], dim=-1)

    return ret


@torch.compile
def unpack_predictions(preds):
    ret = {}
    ret["cls_binary"], ret["cls_id_onehot"], ret["momentum"], ret["ispu"] = preds

    # unpacking
    ret["pt"] = ret["momentum"][..., 0]
    ret["eta"] = ret["momentum"][..., 1]
    ret["sin_phi"] = ret["momentum"][..., 2]
    ret["cos_phi"] = ret["momentum"][..., 3]
    ret["energy"] = ret["momentum"][..., 4]

    # first get the cases where a particle was predicted
    ret["cls_id"] = torch.argmax(ret["cls_binary"], dim=-1)
    # when a particle was predicted, get the particle ID
    ret["cls_id"][ret["cls_id"] == 1] = torch.argmax(ret["cls_id_onehot"], dim=-1)[ret["cls_id"] == 1]

    # get the predicted particle ID
    # ret["cls_id"] = torch.argmax(ret["cls_id_onehot"], dim=-1)

    # particle properties
    ret["phi"] = torch.atan2(ret["sin_phi"], ret["cos_phi"])
    p4_tensor_list = [
        ret["pt"].unsqueeze(dim=-1),
        ret["eta"].unsqueeze(dim=-1),
        ret["phi"].unsqueeze(dim=-1),
        ret["energy"].unsqueeze(dim=-1),
    ]
    ret["p4"] = torch.cat(p4_tensor_list, dim=-1)

    return ret


def save_HPs(config: MLPFConfig, mlpf, outdir):
    """Simple function to store the model parameters and training hyperparameters."""

    with open(f"{outdir}/model_kwargs.pkl", "wb") as f:
        pkl.dump(config, f, protocol=pkl.HIGHEST_PROTOCOL)

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)

    with open(f"{outdir}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        outdict = {"num_mlpf_params": num_mlpf_parameters}
        outdict.update(config.model_dump(mode="json"))
        json.dump(outdict, fp)


def get_model_state_dict(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()


def print_optimizer_stats(optimizer, stage):
    print(f"\nOptimizer statistics {stage}:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Parameter group {i}:")
        print(f"    Learning rate: {param_group['lr']}")
        print(f"    Weight decay: {param_group['weight_decay']}")
        if "momentum" in param_group:
            print(f"    Momentum: {param_group['momentum']}")
        elif "betas" in param_group:
            print(f"    Betas: {param_group['betas']}")

    if hasattr(optimizer, "state"):
        print("  Optimizer state:")
        print(f"    Number of steps: {len(optimizer.state)}")
        if len(optimizer.state) > 0:
            first_param = next(iter(optimizer.state.values()))
            for key, value in first_param.items():
                if torch.is_tensor(value):
                    print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {value}")


def load_checkpoint(checkpoint, model, optimizer, strict=True, start_step=0):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if strict:
        print_optimizer_stats(optimizer, "Before loading optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Loaded optimizer state")
        print_optimizer_stats(optimizer, "After loading optimizer state")

    if "rng_state" in checkpoint["extra_state"]:
        torch.set_rng_state(checkpoint["extra_state"]["rng_state"].cpu())
        logging.info("Loaded RNG state")

    return model, optimizer


def save_checkpoint(checkpoint_path, model, optimizer=None, extra_state=None):
    if extra_state is None:
        extra_state = {}
    extra_state["rng_state"] = torch.get_rng_state()
    torch.save(
        {
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "extra_state": extra_state,
        },
        checkpoint_path,
    )


def load_lr_schedule(lr_schedule, checkpoint, start_step=0):
    """Loads the lr_schedule's state dict from checkpoint and sets the last_epoch to start_step"""
    if "lr_schedule_state_dict" in checkpoint["extra_state"].keys():
        state_dict = checkpoint["extra_state"]["lr_schedule_state_dict"]
        lr_schedule.load_state_dict(state_dict)
        lr_schedule.last_epoch = start_step
    else:
        raise KeyError("Couldn't find LR schedule state dict in checkpoint. extra_state contains: {}".format(checkpoint["extra_state"].keys()))


def get_lr_schedule(config: MLPFConfig, opt, num_steps, last_batch=-1):
    if config.lr_schedule == LRSchedule.CONSTANT:
        lr_schedule = ConstantLR(opt, factor=1.0, total_iters=num_steps)
    elif config.lr_schedule == LRSchedule.ONECYCLE:
        lr_schedule = OneCycleLR(
            opt,
            max_lr=config.lr,
            total_steps=num_steps,
            last_epoch=last_batch,
            pct_start=config.lr_schedule_config.get("onecycle", {}).get("pct_start") or 0.3,
        )
    elif config.lr_schedule == LRSchedule.COSINEDECAY:
        lr_schedule = CosineAnnealingLR(opt, T_max=num_steps, last_epoch=last_batch, eta_min=config.lr * 0.1)
    elif config.lr_schedule == LRSchedule.REDUCE_LR_ON_PLATEAU:
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("mode", "min"),
            factor=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("factor", 0.1),
            patience=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("patience", 10),
            threshold=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("threshold", 1e-4),
            threshold_mode=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("threshold_mode", "rel"),
            cooldown=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("cooldown", 0),
            min_lr=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("min_lr", 0),
            eps=config.lr_schedule_config.get("reduce_lr_on_plateau", {}).get("eps", 1e-8),
        )
    else:
        raise ValueError(f"Supported values for lr_schedule are {list(LRSchedule)}")
    return lr_schedule


def count_parameters(model):
    column_names = ["Modules", "Trainable parameters", "Non-trainable parameters"]
    table = pd.DataFrame(columns=column_names)
    trainable_params = 0
    nontrainable_params = 0
    for ii, (name, parameter) in enumerate(model.named_parameters()):
        params = parameter.numel()
        if not parameter.requires_grad:
            table = pd.concat(
                [
                    table,
                    pd.DataFrame({column_names[0]: name, column_names[1]: 0, column_names[2]: params}, index=[ii]),
                ]
            )
            nontrainable_params += params
        else:
            table = pd.concat(
                [
                    table,
                    pd.DataFrame({column_names[0]: name, column_names[1]: params, column_names[2]: 0}, index=[ii]),
                ]
            )
            trainable_params += params
    return trainable_params, nontrainable_params, table
