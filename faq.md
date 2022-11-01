- is the set of inputs for MLPF identical to standard PF?
  - As far as we can tell, to a large extent, yes. We use all features that seem to be meaningful from PFElements. It's not very easy to tell from standard PF what exactly are the inputs, as the inputs are not defined in a single place, but throughout thousands of lines of PF code.

- for applying the focal loss, is there a way to do this for a softmax? I only really see how this works for a set of sigmoid outputs
  - we use the `tfa.losses.sigmoid_focal_crossentropy` function on softmax outputs. It would be good to check that there is no error in how we are using this.
  
- for the object condensation, how do choose which reconstructed object should be assigned the truth label and which ones should be labeled 0 when there are multiple signals?
  - In general, there are many-to-many energy deposits from truth particles to input elements. Truth particles are assigned to a primary input element according to the largest energy deposit, removing that input element for subsequent assignments. In practice, it's always completely straightforward for track-based particles (ch.had, ele, mu), and for calo-based particles, we can see a good correspondence between the truth particle and PFElement (eta,phi) based on this assignment.

- do I understand correctly that from the LSH+kNN step you end up with some large number of smaller graphs, and they are not connected to each other, only internally?
  - Correct. Since we have multiple layers of this graph building + GCN steps, in principle, there can still be connectivity across all elements, just in different layers.
  
- for the study of adding additional physics terms to the loss, was the hyperparameter scan redone?
  - It was not redone. Most likely the story on the additional physics terms is not yet fully concluded, and we might have to revisit it with the new inputs we have received since.

- is there something that happens for track pt>50 GeV? theres a pretty sizable jump in fake rate for CHs there
  - Not that I'm aware of, but it would be good to look into this.

- for the jets resolution performance plots, are these response corrected?
  - after a long discussion with the PF conveners, we concluded that due to the way I made the fits, these response and resolution plots we are showing should be interpreted not as absolute measurements, but rather only as comparisons between PF and MLPF. In future iterations, we will do more detailed studies of these physics plots using the PF tools for response/resolution measurements.

- do we know what the expected runtime scaling vs. # of PFElements is for standard PF?
  - We haven't checked this directly, but standard PF uses KD-trees for the element-to-element linking step, so it should be fairly good. However, there was a long-standing feature/bug in PF linking where it essentially produced two large blocks of elements, one in the forward region, one in the backward region. So an event with N elements would have roughly two blocks of N/2 elements each, making the scaling poor. This has since been fixed. For Phase 2 where this scaling really starts to matter, the PF+HGCAL reco interplay is still open, so the standard PF algo has not been tested in those conditions AFAIK.
