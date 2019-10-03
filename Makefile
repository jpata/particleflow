ntuples:
	#./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValNuGun/pfvalidation/190924_174634/0000/ ./data/NuGun
	#./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValQCD_FlatPt_15_3000HS_13/pfvalidation/190924_174512/0000/ ./data/QCD
	./test/run_ntuple.sh /mnt/hadoop/store/user/jpata/RelValTTbar_13/pfvalidation/190925_042050/0000/ ./data/TTbar

cache:
	\ls -1 ./data/TTbar/*.root | xargs -n 1 -P 8 singularity exec --nv -B /storage /storage/group/gpu/software/singularity/ibanks/edge.simg python3 test/cache.py	
