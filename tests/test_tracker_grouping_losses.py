from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mlpf.conf import SetDecoderConfig, Y_FEATURES
from mlpf.model.set_losses import hit_grouping_loss, tracker_overlap_loss


def fixture():
    torch.manual_seed(7)
    x=torch.zeros(1,8,15);x[0,:6,0]=1;x[0,6,0]=2
    y=torch.zeros(1,8,len(Y_FEATURES));target=torch.zeros(1,3,len(Y_FEATURES))
    pn=Y_FEATURES.index('particle_number')
    y[0,:,pn]=torch.tensor([11,11,11,22,33,0,11,11])
    target[0,:,pn]=torch.tensor([11,22,33]);target[0,:,0]=torch.tensor([1,5,2])
    batch=SimpleNamespace(X=x,mask=x[...,0]!=0,ytarget=y,ytarget_set=target,target_mask=target[...,0]!=0)
    grouping=(torch.randn(1,8,3).mul(.2).requires_grad_(),torch.randn(1,3,3).mul(.2).requires_grad_(),torch.randn(1,8,1).mul(.2).requires_grad_())
    matches=[(torch.tensor([2,0,1]),torch.tensor([0,1,2]))]
    return grouping,batch,matches


def dense_probabilities(g):
    k,s,b=g
    return torch.cat([10*k[0]@s[0].T,b[0]],-1).softmax(-1)


@pytest.mark.parametrize('chunk_size',[1,2,20])
def test_overlap_matches_dense_value_and_gradients(chunk_size):
    g,b,m=fixture()
    result=tracker_overlap_loss(g,b,m,chunk_size)
    assert result['count'].item()==2
    result['loss'].backward()
    gradients=[x.grad.clone() for x in g]
    ref=tuple(x.detach().clone().requires_grad_() for x in g)
    p=dense_probabilities(ref)[:6,:3]
    tp=torch.stack([p[3,0],p[:3,2].sum()]);mass=p.sum(0)[[0,2]];truth=p.new_tensor([1,3])
    expected=(1-(2*tp+1e-6)/(truth+mass+1e-6)).mean()
    torch.testing.assert_close(result['loss'],expected)
    torch.testing.assert_close(result['purity'],(tp/mass).mean())
    torch.testing.assert_close(result['completeness'],(tp/truth).mean())
    expected.backward()
    for actual,x in zip(gradients,ref):torch.testing.assert_close(actual,x.grad,atol=1e-6,rtol=1e-5)
    assert gradients[0][0,6:].count_nonzero()==0  # calo and padding never enter Dice


@pytest.mark.parametrize('detector_balanced',[False,True])
def test_particle_balancing_preserves_foreground_mass_and_other_weights(detector_balanced):
    g,b,m=fixture()
    actual=hit_grouping_loss(g,b,m,chunk_size=2,tracker_particle_balanced=True,detector_balanced=detector_balanced)
    weights=torch.tensor([2/3,2/3,2/3,2,1,.1,1])
    assert weights.sum().item()==pytest.approx(6.1)
    if detector_balanced:
        weights[:6]/=5.1;weights/=2
    else:weights/=6.1
    logits=torch.cat([10*g[0][0,:7]@g[1][0].T,g[2][0,:7]],-1)
    expected=(F.cross_entropy(logits,torch.tensor([2,2,2,0,1,3,2]),reduction='none')*weights).sum()
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert all(torch.isfinite(v.grad).all() for v in g)


def test_disabled_balancing_is_original_hit_average():
    g,b,m=fixture()
    logits=torch.cat([10*g[0][0,:7]@g[1][0].T,g[2][0,:7]],-1)
    weights=torch.tensor([1,1,1,1,1,.1,1])
    expected=(F.cross_entropy(logits,torch.tensor([2,2,2,0,1,3,2]),reduction='none')*weights).sum()/weights.sum()
    torch.testing.assert_close(hit_grouping_loss(g,b,m),expected)


def test_missing_hits_remain_supervised_when_background_wins():
    g,b,m=fixture()
    g=(g[0],g[1],torch.full_like(g[2],6.,requires_grad=True))
    result=tracker_overlap_loss(g,b,m,chunk_size=2)
    assert result['count']==2 and result['loss']>.9
    result['loss'].backward()
    # Lower background logits for missed truth hits under gradient descent.
    assert (g[2].grad[0,:4]>0).all()


@pytest.mark.parametrize('empty',['neutral','no_tracker','no_matches'])
def test_no_eligible_tracks_gives_connected_zero(empty):
    g,b,m=fixture()
    if empty=='neutral':b.ytarget_set[...,0]=2
    if empty=='no_tracker':b.X[0,:6,0]=2
    if empty=='no_matches':m=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]
    result=tracker_overlap_loss(g,b,m,chunk_size=2)
    assert result['count']==0 and result['loss']==0
    result['loss'].backward()
    assert all(v.grad is not None and v.grad.count_nonzero()==0 for v in g)


def test_bfloat16_autocast_keeps_overlap_reductions_float32():
    g,b,m=fixture()
    with torch.autocast('cpu',dtype=torch.bfloat16):result=tracker_overlap_loss(g,b,m,chunk_size=2)
    assert result['loss'].dtype==torch.float32
    result['loss'].backward()
    assert all(torch.isfinite(v.grad).all() for v in g)


@pytest.mark.parametrize('kwargs',[{'hit_grouping_tracker_particle_balanced':True},{'tracker_overlap_loss_weight':.05}])
def test_new_objectives_require_grouping(kwargs):
    with pytest.raises(ValueError,match='require hit_grouping'):SetDecoderConfig(**kwargs)


def test_negative_overlap_weight_is_rejected():
    with pytest.raises(ValueError):SetDecoderConfig(tracker_overlap_loss_weight=-1)
