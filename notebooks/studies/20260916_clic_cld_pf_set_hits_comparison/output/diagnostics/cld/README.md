# Slot-level diagnostics (1000 events)

- presence calibration error (ECE, 20 bins): 0.125
- default threshold 0.5: F1 0.496, efficiency 0.473, purity 0.522, duplicates 0.216, count bias -9.03, MET error 9.00 GeV
- kinematics-limited oracle (all-slot Hungarian accepted): F1 0.801, efficiency 0.668, purity 0.999: the ceiling any selection rule can reach with these slot kinematics
- best plain threshold by F1: 0.70 (F1 0.597, count bias -38.63)
- threshold 0.5 + NMS dR<0.05: F1 0.525, duplicates 0.130
- missed targets with a slot seed hit within dR<0.1: 0.489; missed targets that are track-associated: 0.277
- fraction of Hungarian-matched slots that the decoder moved closer to their target than the seed hit: 0.928
- per decoder layer F1: L1 0.165, L2 0.192, L3 0.215, L4 0.230, L5 0.269, L6 0.361, L7 0.451, L8 0.496

Figures: presence_calibration.png, selection_policies.png, query_types.png, reference_refinement.png, per_layer.png, slot_budget.png.
Tables: presence_calibration.csv, selection_policies.csv, query_types.csv, reference_refinement.csv, per_layer.csv, slot_budget.csv.
