## **TLDR; I just want to run the code**
```
pip install -r requirements.txt
./scripts/local_test_torch.sh
```

### **Summary**

**ML-based particle flow (MLPF)** focuses on developing full event reconstruction for particle detectors using computationally scalable and flexible machine learning models. The project aims to improve particle flow reconstruction across various detector environments, including CMS, as well as future detectors via Key4HEP.
We build on existing, open-source simulation software by the experimental collaborations.

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
| [3.0.0](https://github.com/jpata/particleflow/releases/v3.0.0) | 3.0.0 | 3.0.0 | 3.0.0 |

---

## **Getting Started with Pixi & Snakemake**

The full data generation, model training, and validation workflow are managed using [Pixi](https://pixi.sh/) for environment management and [Snakemake](https://snakemake.readthedocs.io/) for job orchestration.

### **1. Install Pixi**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
# Restart your shell or source your .bashrc
```

### **2. Initialize Your Site**
Configure the environment for your specific cluster. This sets up the necessary Snakemake profiles and site defaults.

*   **Tallinn (Slurm):**
```bash
pixi run -e tallinn init
```
*   **lxplus (HTCondor):**
```bash
pixi run -e lxplus init
```

### **3. Generate the Workflow**
Generate the `Snakefile` for a production campaign corresponding to your site.
```bash
PROD=cms_run3 STEPS=gen,post,tfds,train pixi run -e lxplus generate
```
You can inspect `snakemake_jobs/cms_run3/Snakefile` and the related scripts to understand the workflow.

### **4. Execute the Workflow**
Launch the workflow on the batch system. It is recommended to run this inside a `tmux` or `screen` session.
```bash
PROD=cms_run3 STEPS=gen,post,tfds,train pixi run -e lxplus run
```

### **5. Validation & Plots**
To run the validation plotting workflow:
```bash
PROD=cms_run3 pixi run -e lxplus validation
```

---

# **Citations and Reuse**

You are welcome to reuse the code in accordance with the [LICENSE](https://github.com/jpata/particleflow/blob/main/LICENSE).

**How to Cite**

1. **Academic Work:** Please cite the specific papers listed in the **Publications** section above relevant to the method you are using (e.g., initial GNN idea, fine-tuning, or specific detector studies).
2. **Code Usage:** If you use the code significantly for research, please cite the specific [tagged version from Zenodo](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version).
3. **Dataset Usage:** Cite the [appropriate dataset](https://zenodo.org/search?q=mlpf&f=allversions%3Atrue&f=resource_type%3Adataset&l=list&p=1&s=10&sort=version) via the Zenodo link and the corresponding paper.

**Contact**

For collaboration ideas that do not fit into the categories above, please [get in touch via GitHub Discussions](https://github.com/jpata/particleflow/discussions/categories/general).
