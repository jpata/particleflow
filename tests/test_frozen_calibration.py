import numpy as np
from scripts.studies.calibrate_frozen_tracker_readout import design, fit_controls, calibration, repeated_quantile


def test_calibration_recovers_known_predicted_feature_relationship():
    rng=np.random.default_rng(2)
    base=np.zeros((4,64,5));base[...,0]=1;base[...,1]=rng.uniform(-2,4,(4,64));base[...,2]=rng.uniform(-3,3,(4,64))
    beta=np.array([.02,-.03,.01,.005,.004,-.002])
    truth=base.copy();truth[...,1]+=design(base,'polynomial')@beta
    cache={'base':base,'truth':truth,'match_index':np.tile(np.arange(64),(4,1))}
    fitted=fit_controls(cache,np.ones((4,64)),'polynomial',[0.])[0]
    np.testing.assert_allclose(fitted['coefficients'],beta,atol=1e-10)
    np.testing.assert_allclose(calibration(cache,np.ones((4,64)),fitted),truth[...,1]-base[...,1],atol=1e-10)


def test_calibration_respects_neutral_and_support_gates():
    base=np.zeros((1,3,5));base[0,:,0]=[1,2,1]
    model={'kind':'constant','coefficients':[.2]}
    np.testing.assert_allclose(calibration({'base':base},np.array([[1,1,0]]),model),[[.2,0,0]])


def test_bootstrap_weighted_quantiles_match_explicit_event_resampling():
    values=np.array([-2.,-.1,.2,.4,1.1]);weights=np.array([0,3,0,2,4])
    expanded=np.repeat(values,weights)
    for q in [.16,.5,.84]:
        np.testing.assert_allclose(repeated_quantile(values,weights,q),np.quantile(expanded,q))
