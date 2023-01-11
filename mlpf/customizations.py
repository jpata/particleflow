#
# Functions to customize the config based on commandline flags
#
# Used to make the github pipeline fast
def customize_pipeline_test(config):

    # don't use dynamic batching, as that can result in weird stuff with very few events
    config["batching"]["bucket_by_sequence_length"] = False

    # for cms, keep only ttbar
    if "cms_pf_ttbar" in config["datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["cms_pf_ttbar"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 2
        config["validation_dataset"] = "cms_pf_ttbar"
        config["evaluation_datasets"] = {"cms_pf_ttbar": {"batch_size": 2, "num_events": -1}}

    # For CLIC, keep only ttbar
    if "clic_ttbar_pf" in config["datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["clic_ttbar_pf"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 50
        config["validation_dataset"] = "clic_ttbar_pf"
        config["evaluation_datasets"] = {"clic_ttbar_pf": {"batch_size": 50, "num_events": -1}}

    # validate only on a small number of events
    config["validation_num_events"] = config["validation_batch_size"] * 2

    return config


# Register all the customization functions
customization_functions = {"pipeline_test": customize_pipeline_test}
