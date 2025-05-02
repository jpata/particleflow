#!/bin/bash

#sbatch mlpf/tallinn/postprocessing.sh p8_ee_tt_ecm380
#sbatch mlpf/tallinn/postprocessing.sh p8_ee_qq_ecm380
#sbatch mlpf/tallinn/postprocessing.sh p8_ee_tt_ecm380_PU10
#sbatch mlpf/tallinn/postprocessing.sh p8_ee_WW_fullhad_ecm380
#sbatch mlpf/tallinn/postprocessing.sh p8_ee_ZH_Htautau_ecm380

#sbatch mlpf/tallinn/postprocessing_hits.sh p8_ee_tt_ecm380
#sbatch mlpf/tallinn/postprocessing_hits.sh p8_ee_qq_ecm380
#sbatch mlpf/tallinn/postprocessing_hits.sh kaon0L
#sbatch mlpf/tallinn/postprocessing_hits.sh pi-
#sbatch mlpf/tallinn/postprocessing_hits.sh pi+
#sbatch mlpf/tallinn/postprocessing_hits.sh pi0
#sbatch mlpf/tallinn/postprocessing_hits.sh e-
#sbatch mlpf/tallinn/postprocessing_hits.sh e+
#sbatch mlpf/tallinn/postprocessing_hits.sh mu-
#sbatch mlpf/tallinn/postprocessing_hits.sh mu+
#sbatch mlpf/tallinn/postprocessing_hits.sh gamma
#sbatch mlpf/tallinn/postprocessing_hits.sh neutron
