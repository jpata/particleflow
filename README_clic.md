Download the .slcio files from https://atlaswww.hep.anl.gov/hepsim/list.php?find=rfull201 -> `gev380ee_pythia6_ttbar` to

```
data/clic/gev380ee_pythia6_ttbar_rfull201/root/
```

Run
```bash
ls data/clic/gev380ee_pythia6_ttbar_rfull201/root/*.slcio | parallel -j4 singularity exec delphes/centos7hepsim.sif ./clic/process_data.sh {}
```
