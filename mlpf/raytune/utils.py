from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandForBOHB,
    HyperBandScheduler,
    PopulationBasedTraining,
)
from ray.tune.schedulers.pb2 import PB2  # Population Based Bandits
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.skopt import SkOptSearch

# from ray.tune.search.hebo import HEBOSearch # HEBO is not yet supported


def get_raytune_search_alg(raytune_cfg, seeds=False):
    if (raytune_cfg["sched"] == "pbt") or (raytune_cfg["sched"] == "pb2"):
        if raytune_cfg["search_alg"] is not None:
            print(
                "INFO: Using schedule '{}' is not compatible with Ray Tune search algorithms.".format(raytune_cfg["sched"])
            )
            print("INFO: Uing the Ray Tune {} scheduler without search algorithm".format(raytune_cfg["sched"]))
        return None

    if (raytune_cfg["sched"] == "bohb") or (raytune_cfg["sched"] == "BOHB"):
        print("INFO: Using TuneBOHB search algorithm since it is required for BOHB shedule")
        if seeds:
            seed = 1234
        else:
            seed = None
        return TuneBOHB(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            seed=seed,
        )

    # requires pip install bayesian-optimization
    if raytune_cfg["search_alg"] == "bayes":
        print("INFO: Using BayesOptSearch")
        return BayesOptSearch(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            random_search_steps=raytune_cfg["bayes"]["n_random_steps"],
        )

    # requires pip install hyperopt
    if raytune_cfg["search_alg"] == "hyperopt":
        print("INFO: Using HyperOptSearch")
        return HyperOptSearch(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            n_initial_points=raytune_cfg["hyperopt"]["n_random_steps"],
            # points_to_evaluate=,
        )
    if raytune_cfg["search_alg"] == "scikit":
        print("INFO: Using bayesian optimization from scikit-learn")
        return SkOptSearch(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            convert_to_python=True,
        )
    # HEBO is not yet supported
    # if (raytune_cfg["search_alg"] == "hebo") or (raytune_cfg["search_alg"] == "HEBO"):
    #     print("Using HEBOSearch")
    #     return HEBOSearch(
    #         metric=raytune_cfg["default_metric"],
    #         mode=raytune_cfg["default_mode"],
    #         # max_concurrent=8,
    #     )
    else:
        print("INFO: Not using any Ray Tune search algorithm")
        return None


def get_raytune_schedule(raytune_cfg):
    if raytune_cfg["sched"] == "asha":
        return AsyncHyperBandScheduler(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            time_attr="training_iteration",
            max_t=raytune_cfg["asha"]["max_t"],
            grace_period=raytune_cfg["asha"]["grace_period"],
            reduction_factor=raytune_cfg["asha"]["reduction_factor"],
            brackets=raytune_cfg["asha"]["brackets"],
        )
    elif raytune_cfg["sched"] == "hyperband":
        return HyperBandScheduler(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            time_attr="training_iteration",
            max_t=raytune_cfg["hyperband"]["max_t"],
            reduction_factor=raytune_cfg["hyperband"]["reduction_factor"],
        )
    # requires pip install hpbandster ConfigSpace
    elif (raytune_cfg["sched"] == "bohb") or (raytune_cfg["sched"] == "BOHB"):
        return HyperBandForBOHB(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            time_attr="training_iteration",
            max_t=raytune_cfg["hyperband"]["max_t"],
            reduction_factor=raytune_cfg["hyperband"]["reduction_factor"],
        )
    elif (raytune_cfg["sched"] == "pbt") or (raytune_cfg["sched"] == "PBT"):
        return PopulationBasedTraining(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            time_attr="training_iteration",
            perturbation_interval=raytune_cfg["pbt"]["perturbation_interval"],
            hyperparam_mutations=raytune_cfg["pbt"]["hyperparam_mutations"],
            log_config=True,
        )
    # requires pip install GPy sklearn
    elif (raytune_cfg["sched"] == "pb2") or (raytune_cfg["sched"] == "PB2"):
        return PB2(
            metric=raytune_cfg["default_metric"],
            mode=raytune_cfg["default_mode"],
            time_attr="training_iteration",
            perturbation_interval=raytune_cfg["pb2"]["perturbation_interval"],
            hyperparam_bounds=raytune_cfg["pb2"]["hyperparam_bounds"],
            log_config=True,
        )
    else:
        print("INFO: Not using any Ray Tune trial scheduler.")
        return None
