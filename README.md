### **Summary**

**ML-based particle flow (MLPF)** focuses on developing full event reconstruction for particle detectors using computationally scalable and flexible machine learning models. The project aims to improve particle flow reconstruction across various detector environments, including CMS, as well as future detectors via Key4HEP.
We build on existing, open-source simulation software by the experimental collaborations.

Read the [user documentation](https://jpata.github.io/particleflow/) for the physics overview, installation and workflow guides, current capabilities, publications, and roadmap. The Markdown source is in [`docs/`](docs/index.md).

<p float="left">
  <img src="images/diagram.svg" alt="High-level overview" width="800"/>
</p>

---

### **TLDR; I just want to run the code**
You can use `uv` to set up the repo and test that everything works:
```
git clone --recurse-submodules https://github.com/jpata/particleflow.git
uv sync --project envs/ort-gpu
uv run ./scripts/local_test_cld.sh
uv run ./scripts/local_test_cms.sh
```

Alternatively, you can use a prepared container:
```
apptainer exec --nv https://jpata.web.cern.ch/jpata/pytorch-20260305-08d6950.sif ./scripts/local_test_cld.sh
apptainer exec --nv https://jpata.web.cern.ch/jpata/pytorch-20260305-08d6950.sif ./scripts/local_test_cms.sh
```



### **Datasets**

See the user documentation to [choose a dataset](https://jpata.github.io/particleflow/datasets/catalog), [download and verify published TFDS data](https://jpata.github.io/particleflow/datasets/download), or [produce and publish a dataset](https://jpata.github.io/particleflow/datasets/generate).

### **Training**

The [training guide](https://jpata.github.io/particleflow/training/train) provides a verified CPU check, a bounded GPU example, expected outputs, and checkpoint continuation.

### **Model Upload**

To upload a trained model to the Hugging Face Hub:
```bash
uv run python3 scripts/upload_model_hf.py experiments/pyg-clic-hits-v1_clic_20260328_144021_479374 --version v3.1.0
```

### **Model Download & Evaluation**

The [Key4HEP evaluation guide](https://jpata.github.io/particleflow/validation/key4hep) downloads a published checkpoint and runs it on an example EDM4hep ROOT file. The [validation overview](https://jpata.github.io/particleflow/validation/overview) explains which checks are needed before interpreting physics performance.

## **End-to-end workflow**

The [dataset-generation guide](https://jpata.github.io/particleflow/datasets/generate) documents the Pixi/Snakemake pipeline from detector simulation through validated TFDS output. Training, evaluation, physics validation, and ONNX validation are covered in the [user documentation](https://jpata.github.io/particleflow/).

---

### **Publications**

The following publications trace the development of MLPF from early proofs of concept to full detector simulations and fine-tuning studies across detectors.

* [2021] First full-event GNN demonstration of MLPF: [Paper](https://doi.org/10.1140/epjc/s10052-021-09158-w) [Code](https://zenodo.org/records/4559587) [Dataset](https://doi.org/10.5281/zenodo.4559324)
* [2021] First demonstration in CMS Run 3: [Paper](http://dx.doi.org/10.1088/1742-6596/2438/1/012100) [CMS-DP](https://cds.cern.ch/record/2792320)
* [2022] Improved performance in CMS Run 3: [CMS-DP](http://cds.cern.ch/record/2842375)
* [2024] Improved performance with full simulation for future colliders: [Paper](https://doi.org/10.1038/s42005-024-01599-5) [Code](https://zenodo.org/records/10928968) [Results](https://doi.org/10.5281/zenodo.10567397)
* [2025] Fine-tuning across detectors: [Paper](https://doi.org/10.1103/PhysRevD.111.092015) [Code](https://zenodo.org/records/14930299)
* [2026] CMS Run 3 full results: [Paper](https://arxiv.org/abs/2601.17554) [CMS-DP](https://cds.cern.ch/record/2937578) [Code](https://zenodo.org/records/15573658)

---

### **Citations and Reuse**

You are welcome to reuse the code in accordance with the [LICENSE](https://github.com/jpata/particleflow/blob/main/LICENSE).

**How to Cite**

1. **Academic Work:** Please cite the specific papers listed in the **Publications** section above relevant to the method you are using (e.g., initial GNN idea, fine-tuning, or specific detector studies).
2. **Code Usage:** If you use the code significantly for research, please cite the specific [tagged version from Zenodo](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version).
3. **Dataset Usage:** Cite the [appropriate dataset](https://zenodo.org/search?q=mlpf&f=allversions%3Atrue&f=resource_type%3Adataset&l=list&p=1&s=10&sort=version) via the Zenodo link and the corresponding paper.

**Contact**

For other collaboration ideas, please [get in touch via GitHub Discussions](https://github.com/jpata/particleflow/discussions/categories/general).
