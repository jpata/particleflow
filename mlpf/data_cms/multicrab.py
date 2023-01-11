import os

from CRABAPI.RawCommand import crabCommand
from CRABClient.UserUtilities import config


def submit(config):
    crabCommand("submit", config=config)
    # save crab config for the future
    with open(
        config.General.workArea + "/crab_" + config.General.requestName + "/crab_config.py",
        "w",
    ) as fi:
        fi.write(config.pythonise_())


# https://cmsweb.cern.ch/das/request?view=plain&limit=50&instance=prod%2Fglobal&input=%2FRelVal*%2FCMSSW_11_0_0_pre4*%2FGEN-SIM-DIGI-RAW
samples = [
    (
        "/RelValQCD_FlatPt_15_3000HS_14/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW",
        "QCD_run3",
    ),
    (
        "/RelValNuGun/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW",
        "NuGun_run3",
    ),
    (
        "/RelValTTbar_14TeV/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW",
        "TTbar_run3",
    ),
    # ("/RelValTTbar_14TeV/CMSSW_11_0_0_pre12-PU25ns_110X_mcRun4_realistic_v2_2026D41PU140-v1/GEN-SIM-DIGI-RAW",
    # "TTbar_run4_pu140"),
    # ("/RelValTTbar_14TeV/CMSSW_11_0_0_pre12-PU25ns_110X_mcRun4_realistic_v2_2026D41PU200-v1/GEN-SIM-DIGI-RAW",
    # "TTbar_run4_pu200")
]

if __name__ == "__main__":
    for dataset, name in samples:

        if os.path.isfile("step3_dump.pyc"):
            os.remove("step3_dump.pyc")

        conf = config()

        conf.General.requestName = name
        conf.General.transferLogs = True
        conf.General.workArea = "crab_projects"
        conf.JobType.pluginName = "Analysis"
        conf.JobType.psetName = "step3_dump.py"
        conf.JobType.maxJobRuntimeMin = 8 * 60
        conf.JobType.allowUndistributedCMSSW = True
        conf.JobType.outputFiles = [
            "step3_inMINIAODSIM.root",
            "step3_AOD.root",
        ]
        conf.JobType.maxMemoryMB = 6000
        conf.JobType.numCores = 2

        conf.Data.inputDataset = dataset
        conf.Data.splitting = "LumiBased"
        conf.Data.unitsPerJob = 2
        # conf.Data.totalUnits = 50
        conf.Data.publication = False
        conf.Data.outputDatasetTag = "pfvalidation"
        # conf.Data.ignoreLocality = True

        # Where the output files will be transmitted to
        # conf.Site.storageSite = 'T3_US_Baylor'
        conf.Site.storageSite = "T2_US_Caltech"
        # conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN"]

        submit(conf)
