### **Summary**

**ML-based particle flow (MLPF)** focuses on developing full event reconstruction for particle detectors using computationally scalable and flexible machine learning models. The project aims to improve particle flow reconstruction across various detector environments, including CMS, as well as future detectors via Key4HEP.
We build on existing, open-souce simulation software by the experimental collaborations.

<p float="left">
  <img src="images/schematic.png" alt="High-level overview" width="600"/>
</p>

---

### **Publications**

Below is the development timeline of MLPF by our team, ranging from initial proofs of concept to full detector simulations and fine-tuning studies.

**2021: First full-event GNN demonstration of MLPF**
* **Paper:** [MLPF: efficient machine-learned particle-flow reconstruction using graph neural networks](https://doi.org/10.1140/epjc/s10052-021-09158-w) (Eur. Phys. J. C)
* **Focus:** Initial idea with a GNN and scalable graph building.
* **Code:** [v1.1](https://zenodo.org/records/4559587)
* **Dataset:** [Zenodo Record](https://doi.org/10.5281/zenodo.4559324)

**2021: First demonstration in CMS Run 3**
* **Paper:** [Machine Learning for Particle Flow Reconstruction at CMS](http://dx.doi.org/10.1088/1742-6596/2438/1/012100) (J. Phys. Conf. Ser.)
* **Focus:** First demonstration of feasibility within CMS.
* **Detector Performance Note:** [CERN-CMS-DP-2021-030](https://cds.cern.ch/record/2792320)

**2022: Improved performance in CMS Run 3**
* **Detector Performance Note:** [CERN-CMS-DP-2022-061](http://cds.cern.ch/record/2842375)
* **Focus:** We showed that training against a generator-level target can improve performance in CMS.

**2024: Improved performance with full simulation for future colliders**
* **Paper:** [Improved particle-flow event reconstruction with scalable neural networks for current and future particle detectors](https://doi.org/10.1038/s42005-024-01599-5) (Communications Physics)
* **Focus:** Improved event-level performance in full simulation for future colliders.
* **Code:** [v1.6.2](https://zenodo.org/records/10928968)
* **Results:** [Zenodo Record](https://doi.org/10.5281/zenodo.10567397)

**2025: Fine-tuning across detectors**
* **Paper:** [Fine-tuning machine-learned particle-flow reconstruction for new detector geometries in future colliders](https://doi.org/10.1103/PhysRevD.111.092015) (Phys. Rev. D)
* **Focus:** Showing that the amount of training data can be reduced by 10x by fine-tuning.
* **Code:** [v2.3.0](https://zenodo.org/records/14930299)

**2026: CMS Run 3 full results**
* **Detector Performance Note:** [CERN-CMS-DP-2025-033](https://cds.cern.ch/record/2937578)
* **Focus:** Improve jet performance over baseline, first validation on real data.
* **Paper:** [CMS Run 3 paper](https://arxiv.org/abs/2601.17554) (submitted to EPJC)
* **Code:** [v2.4.0](https://zenodo.org/records/15573658)

---

### **Datasets**

#### **Software & Dataset Compatibility**

Please ensure you use the correct version of the `jpata/particleflow` software with the corresponding dataset version.

| Code Version | CMS Dataset | CLIC Dataset | CLD Dataset |
| --- | --- | --- | --- |
| [1.9.0](https://github.com/jpata/particleflow/releases/v1.9.0) | 2.4.0 | 2.2.0 | NA |
| [2.0.0](https://github.com/jpata/particleflow/releases/v2.0.0) | 2.4.0 | 2.3.0 | NA |
| [2.1.0](https://github.com/jpata/particleflow/releases/v2.1.0) | 2.5.0 | 2.5.0 | NA |
| [2.2.0](https://github.com/jpata/particleflow/releases/v2.2.0) | 2.5.0 | 2.5.0 | 2.5.0 |
| [2.3.0](https://github.com/jpata/particleflow/releases/v2.3.0) | 2.5.0 | 2.5.0 | 2.5.0 |
| [2.4.0](https://github.com/jpata/particleflow/releases/v2.4.0) | 2.6.0 | 2.5.0 | 2.5.0 |

---

## Running the workflow with Snakemake
The full data generation, model training and validation workflow can be managed using [Snakemake](https://snakemake.readthedocs.io/). Snakemake must be available on both the interactive and worker nodes. The examples below demonstrate how to test it on lxplus.

### 1. Configure for your site

Change `particleflow_spec.yaml` to suit your site, edit the config as needed, for example:
```diff
-  <<: *tallinn
+  <<: *lxplus
```

### 2. Generate the Snakefile
Use the provided script to generate a `Snakefile` for a specific production campaign and model. The following commands create the workflows to run everything from scratch, which requires weeks of time with a few thousand job slots, access to CERN cvmfs and multiple terabytes of storage.
```bash
./scripts/wrapper_lxplus.sh python3 mlpf/produce_snakemake.py --production clic_2025_edm4hep --steps gen,post,tfds,train --model pyg-clic-v1
./scripts/wrapper_lxplus.sh python3 mlpf/produce_snakemake.py --production cld_2025_edm4hep --steps gen,post,tfds,train --model pyg-cld-v1
./scripts/wrapper_lxplus.sh python3 mlpf/produce_snakemake.py --production cms_2025_main --steps gen,post,tfds,train --model pyg-cms-v1
```
You can also produce the snakefile for specific steps only, e.g. `--steps train` to start from existing ML datasets.

### 3. Execute the workflow
Run Snakemake using the generated `Snakefile`. The following example runs locally using a single core, which can be used for debugging.
Note that a super old GPU like P100, V100 or T4 won't work for training! You need at least Ampere generation or newer.
The reference for training is an 80GB A100 GPU.
```bash
./scripts/wrapper_lxplus.sh snakemake -s snakemake_jobs/cld_2025_edm4hep/Snakefile --cores 1 --printshellcmds
```

### 4. Run a large-scale production on batch
Batch systems are different site-by-site. Check some examples below on how to run the full workflow using a batch system:
```bash
./scripts/tallinn/produce_key4hep_snakemake.sh
#TODO: set up and test an example workflow how to run on lxplus condor
```

# Citations and reuse

You are welcome to reuse the code in accordance with the [LICENSE](https://github.com/jpata/particleflow/blob/main/LICENSE).

**How to Cite**

1. **Academic Work:** Please cite the specific papers listed in the **Publications** section above relevant to the method you are using (e.g., initial GNN idea, fine-tuning, or specific detector studies).
2. **Code Usage:** If you use the code significantly for research, please cite the specific [tagged version from Zenodo](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version).
3. **Dataset Usage:** Cite the [appropriate dataset](https://zenodo.org/search?q=mlpf&f=allversions%3Atrue&f=resource_type%3Adataset&l=list&p=1&s=10&sort=version) via the Zenodo link and the corresponding paper.

**Contact**

For collaboration ideas that do not fit into the categories above, please [get in touch via GitHub Discussions](https://github.com/jpata/particleflow/discussions/categories/general).
