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
* **Focus:** Initial idea with a scalable GNN.
* **Code:** [v1.1](https://zenodo.org/records/4559587)
* **Dataset:** [Zenodo Record](https://doi.org/10.5281/zenodo.4559324)

**2021: First demonstration in CMS Run 3**
* **Paper:** [Machine Learning for Particle Flow Reconstruction at CMS](http://dx.doi.org/10.1088/1742-6596/2438/1/012100) (J. Phys. Conf. Ser.)
* **Note:** [CERN-CMS-DP-2021-030](https://cds.cern.ch/record/2792320)

**2022: Improved performance in CMS Run 3**
* **Note:** [CERN-CMS-DP-2022-061](http://cds.cern.ch/record/2842375)

**2024: Improved performance with CLIC full simulation**
* **Paper:** [MLPF: efficient machine-learned particle-flow reconstruction using graph neural networks](https://doi.org/10.1038/s42005-024-01599-5) (Communications Physics)
* **Focus:** Improved event-level performance in full simulation.
* **Code:** [v1.6.2](https://zenodo.org/records/10928968)
* **Results:** [Zenodo Record](https://doi.org/10.5281/zenodo.10567397)

**2025: Fine-tuning across detectors**
* **Paper (Fine-tuning):** [Fine-tuning from CLIC to CLD](https://doi.org/10.1103/PhysRevD.111.092015) (Phys. Rev. D)
* **Code:** [v2.3.0](https://github.com/jpata/particleflow/releases/tag/v2.3.0)

**2025/2026: CMS Run 3 full results**
* **Note (EPS-HEP 2025):** [CERN-CMS-DP-2025-033](https://cds.cern.ch/record/2937578)
* **Paper:** [CMS Run 3 paper](https://arxiv.org/abs/2601.17554) (arXiv, submitted to EPJC)
* **Code:** [v2.4.0](https://github.com/jpata/particleflow/releases/tag/v2.4.0)

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
The full event reconstruction and model training workflow can be managed using [Snakemake](https://snakemake.readthedocs.io/). Snakemake must be available on both the interactive and worker nodes.

### 1. Generate the Snakefile
Use the provided script to generate a `Snakefile` for a specific production campaign and model.
```bash
python3 mlpf/produce_snakemake.py --production clic_2025_edm4hep --steps gen,post,tfds
```

### 2. Execute the workflow
Run Snakemake using the generated `Snakefile`. The following example uses SLURM and Apptainer:
```bash
snakemake -s snakemake_jobs/clic_2025_edm4hep/Snakefile --executor slurm --jobs 100 --use-apptainer
```

To include model training:
```bash
python3 mlpf/produce_snakemake.py --production clic_2025_edm4hep --steps train --model pyg-clic-v1
snakemake -s snakemake_jobs/clic_2025_edm4hep/Snakefile --executor slurm --jobs 1 --use-apptainer --apptainer-args "--nv"
```

# Citations and reuse

You are welcome to reuse the code in accordance with the [LICENSE](https://github.com/jpata/particleflow/blob/main/LICENSE).

**How to Cite**

1. **Academic Work:** Please cite the specific papers listed in the **Publications** section above relevant to the method you are using (e.g., initial GNN idea, fine-tuning, or specific detector studies).
2. **Code Usage:** If you use the code significantly for research, please cite the specific [tagged version from Zenodo](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version).
3. **Dataset Usage:** Cite the [appropriate dataset](https://zenodo.org/search?q=mlpf&f=allversions%3Atrue&f=resource_type%3Adataset&l=list&p=1&s=10&sort=version) via the Zenodo link and the corresponding paper.

**Contact**

For collaboration ideas that do not fit into the categories above, please [get in touch via GitHub Discussions](https://github.com/jpata/particleflow/discussions/categories/general).
