# Slot-level diagnostics (1000 events)

- presence calibration error (ECE, 20 bins): 0.072
- default threshold 0.5: F1 0.617, efficiency 0.605, purity 0.630, duplicates 0.205, count bias -3.77, MET error 8.15 GeV
- kinematics-limited oracle (all-slot Hungarian accepted): F1 0.892, efficiency 0.805, purity 1.000: the ceiling any selection rule can reach with these slot kinematics
- best plain threshold by F1: 0.75 (F1 0.706, count bias -25.27)
- threshold 0.5 + NMS dR<0.05: F1 0.647, duplicates 0.121
- missed targets with a slot seed hit within dR<0.1: 0.513; missed targets that are track-associated: 0.358
- fraction of Hungarian-matched slots that the decoder moved closer to their target than the seed hit: 0.947
- per decoder layer F1: L1 0.216, L2 0.331, L3 0.472, L4 0.617

Figures: presence_calibration.png, selection_policies.png, query_types.png, reference_refinement.png, per_layer.png, slot_budget.png.
Tables: presence_calibration.csv, selection_policies.csv, query_types.csv, reference_refinement.csv, per_layer.csv, slot_budget.csv.
