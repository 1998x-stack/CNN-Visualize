"""V0.5 controlled-measurement, endpoint and defensive correctness tests."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
import torch
from cnnviz import dynamics as dyn
from cnnviz import concepts as cx
from cnnviz.research import run_research
from cnnviz.server import app


def clear_cache():
    for fn in (dyn.load_run,dyn.probe_data,dyn.model_at,dyn.activations_at,dyn.activation_gram):
        fn.cache_clear()

@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    path=tmp_path_factory.mktemp('concepts')
    a=run_research(name='seed11',seed=11,split_seed=42,epochs=2,
          train_limit=128,val_limit=48,probe_count=32,interval=1,out=str(path))
    b=run_research(name='seed22',seed=22,split_seed=42,epochs=2,
          train_limit=128,val_limit=48,probe_count=32,interval=1,out=str(path))
    bad=run_research(name='different-probes',seed=33,split_seed=78,epochs=1,
          train_limit=128,val_limit=48,probe_count=32,interval=1,out=str(path))
    previous=dyn.BASE_DIR;dyn.BASE_DIR=path;clear_cache()
    yield (a,b,bad,TestClient(app))
    dyn.BASE_DIR=previous;clear_cache()


def test_identical_probe_alignment_and_distinct_training(pair):
    a,b,_,_=pair
    x,y=cx._aligned('seed11','seed22')
    assert len(x)==32 and len(y)==32
    assert a['seed']!=b['seed'] and a['split_seed']==b['split_seed']==42
    assert a['probe_indices']==b['probe_indices']
    assert not torch.equal(dyn.model_at('seed11',0).conv1.weight,dyn.model_at('seed22',0).conv1.weight)
    with pytest.raises(ValueError,match='not aligned'):cx._aligned('seed11','different-probes')


def test_topk_order_and_fixed_scale(pair):
    a,b,_,client=pair
    resp=client.get('/api/concepts/topk',params={'run_id':'seed11','step':4,'layer':3,'channel':0,'k':8})
    assert resp.status_code==200,resp.text
    data=resp.json(); scores=[z['peak_response'] for z in data['top_k']]
    assert scores==sorted(scores,reverse=True)
    assert data['pool_size']==32 and len(data['top_k'])==8
    assert sum(data['counts_in_top_k'])==8
    assert all(z['input'].startswith('data:image/png;base64,') for z in data['top_k'])
    assert data['top_k'][0]['activation'].startswith('data:image/png;base64,')
    x,_=dyn.probe_data('seed11');_,features=dyn.activations_at('seed11',4)
    truepeak=features[2][:,0].flatten(1).max(1).values
    assert data['top_k'][0]['probe_index']==int(truepeak.argmax())
    assert client.get('/api/concepts/topk',params={'run_id':'seed11','step':4,'layer':3,'channel':999}).status_code==400


def test_hungarian_matching_and_cross_run_bootstrap(pair):
    _,_,_,client=pair
    a=np.array([[1.,2.,3.,4.],[4.,3.,2.,1.]])
    # Correct response matching recovers a known channel permutation.
    corr=cx._normalized_rows(a)@cx._normalized_rows(a[[1,0]]).T
    row,col=__import__('scipy').optimize.linear_sum_assignment(-corr)
    assert list(col)==[1,0]
    resp=client.get('/api/concepts/compare',params={'run_a':'seed11','run_b':'seed22',
          'step_a':4,'step_b':4,'layer':3,'bootstrap':20})
    assert resp.status_code==200,resp.text
    d=resp.json();assert 0<=d['cka']<=1
    assert len(d['matches'])>0 and d['probe_count']==32
    assert d['bootstrap_ci95'][0]<=d['bootstrap_ci95'][1]
    identical=cx.compare_runs('seed11','seed11',4,4,3,bootstrap=15)
    assert identical['cka']==pytest.approx(1,abs=1e-6)
    bad=client.get('/api/concepts/compare',params={'run_a':'seed11','run_b':'different-probes',
          'step_a':4,'step_b':2,'layer':3})
    assert bad.status_code==400 and 'not aligned' in bad.text


def test_faithfulness_real_proba_and_ood_note(pair):
    _,_,_,client=pair
    d=client.get('/api/concepts/faithfulness',params={'run_id':'seed11','step':4,
       'sample_index':0,'target_class':4}).json()
    assert len(d['fractions'])==len(d['top_cam_deletion'])
    assert d['fractions'][0]==0 and d['fractions'][-1]==1
    assert d['top_cam_deletion'][0]==pytest.approx(d['probability_before'],abs=1e-6)
    assert d['random_deletion_mean'][0]==pytest.approx(d['probability_before'],abs=1e-6)
    assert d['auc_random_minus_cam']==pytest.approx(d['auc_random']-d['auc_top_cam'],abs=3e-6)
    assert 'OOD' in d['note']
    assert client.get('/api/concepts/faithfulness',params={'run_id':'seed11','step':4,'target_class':99}).status_code==400


def test_v05_static_and_prior_routes(pair):
    _,_,_,client=pair
    assert client.get('/concepts.css').status_code==200
    assert client.get('/concepts.js').status_code==200
    assert b'id="concepts"' in client.get('/').content
    assert client.get('/api/dynamics/cka',params={'run_id':'seed11','layer':1}).status_code==200


def test_v05_matched_topk_has_shared_ids(pair):
    _,_,_,client=pair
    q={'run_a':'seed11','run_b':'seed22','step_a':4,'step_b':4,
       'layer':5,'channel_a':0,'channel_b':0,'k':6}
    resp=client.get('/api/concepts/matched-examples',params=q)
    assert resp.status_code==200,resp.text
    d=resp.json()
    assert len(d['topk_a'])==len(d['topk_b'])==6
    assert d['intersection_size']==len(set(x['probe_index'] for x in d['topk_a']) & set(x['probe_index'] for x in d['topk_b']))
    assert 0<=d['jaccard']<=1