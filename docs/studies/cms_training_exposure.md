# CMS training exposure

The `cms_backbone_comparison` scenario and the CMS release checkpoints use larger batches than the v3.2.0 CMS runs. This note explains why.

## What was wrong with v3.2.0

The [v3.2.0 runs](https://huggingface.co/jpata/particleflow/tree/main/cms/v3.2.0) trained with uniform global batches of 64 events for 100k updates. With the recorded dataset sizes, that is only 0.43 passes over each dataset group, and only 9.1% of updates came from the pileup samples.

The [v2.6.0pre1 reference](https://huggingface.co/jpata/particleflow/tree/main/cms/v2.6.0pre1/pyg-cms_20250722_101813_274478) used global batches of 96 events with pileup and 768 without, and reached step 316,400 over ten epochs. Restoring those batches with 320k updates gives about ten passes and 44% pileup updates. Group weights stay unset, so each group is drawn in proportion to its batches per epoch.

## Results so far

Both v3.2.0 models were still improving their validation loss at 100k updates. The published [paired pileup comparison](https://huggingface.co/jpata/particleflow/blob/main/cms/v3.2.0/comparison_metrics/comparison.json) shows better PID accuracy and narrower jet-response IQR than the reference, but worse particle F1 and MET error. Longer training is the next experiment, not evidence that the reference is already surpassed.

## Recipe choices

- Keep the v3.2.0 shared width-512 backbones (three layers, 16 heads) and calibrated task weights. The older width-256 dual backbone and objective differ, so historical aggregate losses are not directly comparable.
- Keep LR 0.0004 with cosine decay to 0.00004. Test the historical 0.002 with `lr-find` before adopting it for these larger models.
- Start fresh runs: resuming a 100k checkpoint restores the old cosine schedule and task weights.
- Validate on 5,000 events per sample; keep the pileup test subsets at 512 per process for comparison with published runs.
- Compare final models on the same held-out inputs and precision, including particle F1/purity, momentum errors, MET, jet response, and angular recall. Narrower response alone can hide a loss of matched jets.

The release checkpoints keep the v3.2.0 model and 100k updates but use these 96/768 batches.
