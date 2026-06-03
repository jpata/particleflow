# Plan: Implementing Object Condensation (OC) in MLPF

This plan outlines the integration of the Object Condensation loss and the `particle_number` target into the MLPF framework.

## 1. Ground Truth & Data Preparation
**Goal:** Ensure every detector element is assigned a unique truth ID for the particle it belongs to.

- [x] **Update Configuration Schema (`mlpf/conf.py`)**:
    - Add `particle_number` to the `ParticleFeatures` dataclass.
- [x] **Generate Unique IDs (`mlpf/data/key4hep/postprocessing.py`)**:
    - In `assign_genparticles_to_obj_and_merge`, assign a continuous integer ID (1..N) to each visible genparticle.
    - Ensure these IDs are propagated to `ytarget_track`, `ytarget_cluster`, and `ytarget_hit`.
- [x] **Target Unpacking (`mlpf/model/utils.py`)**:
    - Verify `unpack_target` correctly identifies `particle_number` (it should happen automatically once added to `ParticleFeatures`).
- [x] **Update Hit-level TFDS Builders (`mlpf/heptfds/cld_pf_edm4hep_hits/`)**:
    - Bump `VERSION` to `3.2.0` in `qq.py`, `ttbar.py`, `ww_fullhad.py`, and `zz.py`.
    - Add release note: `"3.2.0": "Added particle_number target for Object Condensation"`.

## 2. Model Architecture Extensions
**Goal:** Modify the MLPF model to predict the latent space required for clustering.

- [x] **Add OC Output Heads (`mlpf/model/mlpf.py`)**:
    - Add `self.oc_beta`: A Feed-Forward Network (FFN) predicting a single scalar per element.
    - Add `self.oc_coords`: An FFN predicting latent coordinates (e.g., 3D or 4D).
- [x] **Update `MLPF.forward`**:
    - Apply `torch.sigmoid` to the `oc_beta` output to constrain it to $[0, 1]$.
    - Return `preds_oc_beta` and `preds_oc_coords` as part of the model output.
- [x] **Prediction Unpacking (`mlpf/model/utils.py`)**:
    - Update `unpack_predictions` to handle the new return values.

## 3. Loss Function Integration
**Goal:** Implement the math for $L_V$ (clustering) and $L_\beta$ (object detection).

- [x] **Port OC Math (`mlpf/model/losses.py`)**:
    - Adapt `calc_LV_Lbeta` from `HitPF/src/layers/object_cond.py`.
    - Ensure it handles padded batches (MLPF uses 3D tensors `[B, N, D]`).
- [x] **Update `mlpf_loss`**:
    - Add a toggle in config to enable OC loss.
    - Map `particle_number` to `cluster_index_per_event`.
    - Combine $L_V$ and $L_\beta$ into the `Total` loss, potentially replacing or weighting against the standard `Classification_binary` loss.

## 4. Inference & Clustering
**Goal:** Convert model predictions into actual particle candidates.

- [x] **Implement Clustering Logic**:
    - Port `get_clustering` (or a similar heuristic like HDBSCAN) to the inference pipeline.
    - This logic will use high-$\beta$ points as "seeds" and group nearby points in the latent coordinate space.
- [x] **Validation**:
    - Update evaluation scripts to compare OC-reconstructed particles against truth particles.

## 5. Testing Ground Truth
**Goal:** Verify that the `particle_number` is correctly assigned during post-processing.

### Verification Criteria
1. **Uniqueness**: Every unique visible genparticle in an event must be assigned exactly one unique `particle_number` $> 0$.
2. **Consistency**: All detector elements (tracks, clusters, hits) associated with the same genparticle must share the same `particle_number`.
3. **Completeness**: Every detector element matched to a visible genparticle must have a non-zero `particle_number`.
4. **Noise Mapping**: Elements not matched to any visible genparticle must have `particle_number == 0`.
5. **Range**: The set of `particle_number` values in an event must be exactly $\{0, 1, ..., N_{visible\_particles}\}$.

### Test Procedure
1. **Run Post-processing**:
   Run the script on a single sample file from the provided path:
   ```bash
   python mlpf/data/key4hep/postprocessing.py \
     --input /mnt/work/mlpf/cld/v1.2.3_key4hep_2025-05-29_CLD_f1e8f9/gen/p8_ee_ttbar_ecm365/root/p8_ee_ttbar_ecm365_0.root \
     --outpath ./test_out \
     --detector cld \
     --num-events 10
   ```
2. **Inspect Output**:
   Use a notebook or script to load the resulting `.parquet` file and check the `ytarget_track` and `ytarget_cluster` matrices:
   ```python
   import awkward as ak
   data = ak.from_parquet("./test_out/p8_ee_ttbar_ecm365_0.parquet")
   # Check unique IDs in the first event
   particle_ids = data["ytarget_track"][0, :, -1] # Assuming particle_number is last
   print(f"Unique IDs: {set(particle_ids)}")
   ```

## Summary Checklist
- [x] `mlpf/conf.py`: Add `particle_number` to `ParticleFeatures`.
- [x] `mlpf/data/key4hep/postprocessing.py`: Assign unique IDs in `assign_genparticles_to_obj_and_merge`.
- [x] `mlpf/model/mlpf.py`: Add `oc_beta` and `oc_coords` heads.
- [x] `mlpf/model/utils.py`: Update `unpack_predictions` and `unpack_target`.
- [x] `mlpf/model/losses.py`: Integrate `calc_LV_Lbeta` into `mlpf_loss`.
- [ ] **Validation**: Run the test procedure above and verify against the 5 criteria.
