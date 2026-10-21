# Slot-level diagnostics (1000 events)

- presence calibration error (ECE, 20 bins): 0.102
- default threshold 0.5: F1 0.551, efficiency 0.531, purity 0.573, duplicates 0.202, count bias -6.93, MET error 7.39 GeV
- kinematics-limited oracle (all-slot Hungarian accepted): F1 0.837, efficiency 0.720, purity 0.999: the ceiling any selection rule can reach with these slot kinematics
- best plain threshold by F1: 0.70 (F1 0.647, count bias -32.85)
- threshold 0.5 + NMS dR<0.05: F1 0.575, duplicates 0.122
- missed targets with a slot seed hit within dR<0.1: 0.482; missed targets that are track-associated: 0.264
- fraction of Hungarian-matched slots that the decoder moved closer to their target than the seed hit: 0.941
- per decoder layer F1: L1 0.214, L2 0.258, L3 0.300, L4 0.346, L5 0.396, L6 0.465, L7 0.472, L8 0.551

Figures: presence_calibration.png, selection_policies.png, query_types.png, reference_refinement.png, per_layer.png, slot_budget.png.
Tables: presence_calibration.csv, selection_policies.csv, query_types.csv, reference_refinement.csv, per_layer.csv, slot_budget.csv.
