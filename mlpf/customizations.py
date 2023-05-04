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
    if "clic_edm_ttbar_pf" in config["datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["clic_edm_ttbar_pf"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 5
        config["validation_dataset"] = "clic_edm_ttbar_pf"
        config["validation_batch_size"] = 5
        config["evaluation_datasets"] = {"clic_edm_ttbar_pf": {"batch_size": 5, "num_events": -1}}

    if "clic_edm_ttbar_hits_pf" in config["datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["clic_edm_ttbar_hits_pf"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 1
        config["validation_dataset"] = "clic_edm_ttbar_hits_pf"
        config["validation_batch_size"] = 1
        config["evaluation_datasets"] = {"clic_edm_ttbar_hits_pf": {"batch_size": 1, "num_events": -1}}

    # validate only on a small number of events
    config["validation_num_events"] = config["validation_batch_size"] * 2

    config["parameters"]["num_graph_layers_id"] = 1
    config["parameters"]["num_graph_layers_cls"] = 1

    return config


# Register all the customization functions
customization_functions = {"pipeline_test": customize_pipeline_test}
