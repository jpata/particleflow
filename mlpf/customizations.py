#
# Functions to customize the config based on commandline flags
#

# Used to make the github pipeline fast
def customize_pipeline_test(config):

    # don't use dynamic batching, as that can result in weird stuff with very few events
    config["batching"]["bucket_by_sequence_length"] = False

    # for cms, keep only ttbar
    if "physical" in config["train_test_datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["cms_pf_ttbar"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 2
        config["validation_dataset"] = "cms_pf_ttbar"

    # validate only on a small number of events
    config["validation_num_events"] = config["validation_batch_size"] * 2

    return config


# Register all the customization functions
customization_functions = {"pipeline_test": customize_pipeline_test}
