ntuples:
	./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValTTbar_14TeV/pfvalidation/191126_233751/0000/ ./data/TTbar_run3
	./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValQCD_FlatPt_15_3000HS_14/pfvalidation/191126_233511/0000/ ./data/QCD_run3
	./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValNuGun/pfvalidation/191126_233630/0000/ ./data/NuGun_run3


cache:
	\ls -1 ./data/TTbar_run3/*ntuple*.root | xargs -n 1 -P 20 singularity exec -B /storage ~jpata/gpuservers/singularity/images/over_edge.simg python3 test/graph.py
	\ls -1 ./data/QCD_run3/*ntuple*.root | xargs -n 1 -P 20 singularity exec -B /storage ~jpata/gpuservers/singularity/images/over_edge.simg python3 test/graph.py
	\ls -1 ./data/NuGun_run3/*ntuple*.root | xargs -n 1 -P 20 singularity exec -B /storage ~jpata/gpuservers/singularity/images/over_edge.simg python3 test/graph.py

