#
# Functions to customize the config based on commandline flags
#

# Used to make the github pipeline fast
def customize_pipeline_test(config):
    # for cms.yaml, keep only ttbar
    config["batching"]["bucket_by_sequence_length"] = False
    if "physical" in config["train_test_datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["cms_pf_ttbar"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 2
        config["validation_dataset"] = ["cms_pf_ttbar"]

    return config


# Register all the customization functions
customization_functions = {"pipeline_test": customize_pipeline_test}
