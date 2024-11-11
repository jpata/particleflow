#!/bin/bash

rm -f scripts/files_to_copy.txt

maxfiles=100

samplestocopy=(
    "nopu/SingleElectronFlatPt1To1000_pythia8_cfi"
    "nopu/SingleGammaFlatPt1To1000_pythia8_cfi"
    "nopu/SingleK0FlatPt1To1000_pythia8_cfi"
    "nopu/SingleMuFlatPt1To1000_pythia8_cfi"
    "nopu/SingleNeutronFlatPt0p7To1000_cfi"
    "nopu/SinglePi0Pt1To1000_pythia8_cfi"
    "nopu/SinglePiMinusFlatPt0p7To1000_cfi"
    "nopu/SingleProtonMinusFlatPt0p7To1000_cfi"
    "nopu/SingleTauFlatPt1To1000_cfi"
    "nopu/QCDForPF_14TeV_TuneCUETP8M1_cfi"
    "nopu/TTbar_14TeV_TuneCUETP8M1_cfi"
    "nopu/ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi"
    "pu55to75/QCDForPF_14TeV_TuneCUETP8M1_cfi"
    "pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi"
    "pu55to75/ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi"
)

#samplestocopy=(
#    "p8_ee_qq_ecm380"
#    "p8_ee_tt_ecm380"
#    "p8_ee_WW_fullhad_ecm380"
#    "p8_ee_ZH_Htautau_ecm380"
#    "p8_ee_Z_Ztautau_ecm380"
#)


#get a few files from each sample, both the root and postprocessed (raw)
for sample in "${samplestocopy[@]}"; do
    find "/local/joosep/mlpf/cms/./20240823_simcluster/$sample/root" -type f | sort | head -n$maxfiles >> scripts/files_to_copy.txt
    find "/local/joosep/mlpf/cms/./20240823_simcluster/$sample/raw" -type f | sort | head -n$maxfiles >> scripts/files_to_copy.txt
    # find /local/joosep/clic_edm4hep/./2024_07/$sample/root/ -type f | sort | head -n$maxfiles >> scripts/files_to_copy.txt
done

#get the total size
# cat scripts/files_to_copy.txt | xargs du -ch
