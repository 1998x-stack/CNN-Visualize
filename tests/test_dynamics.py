"""Independent V0.4 tests: reference math + genuine trained checkpoints + API."""
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from cnnviz.research import FiveLayerCNN, research_datasets, run_research
from cnnviz import dynamics as dyn
from cnnviz.server import app


def test_zf_alex_layers_and_sampling():
    x=torch.rand(3,1,32,32)
    shapes=[]
    for family in ['zf-small','alex-small']:
        model=FiveLayerCNN(1,10,family)
        logits, features=model.forward_trace(x)
        assert logits.shape==(3,10)
        assert len(features)==5
        assert features[-1].shape[1]==64
        shapes.append(features[0].shape[-1])
    assert shapes==[16,8]  # controlled early stride difference


def test_linear_cka_reference_properties():
    torch.manual_seed(13)
    x=torch.randn(20,12,dtype=torch.float64)
    q,_=torch.linalg.qr(torch.randn(12,12,dtype=torch.float64))
    a=dyn.centered_gram(x)
    b=dyn.centered_gram(x @ q * 3.2)
    assert abs(dyn.linear_cka_gram(a,b)-1)<1e-10
    assert abs(dyn.linear_cka_gram(b,a)-1)<1e-12
    assert dyn.linear_cka_gram(a,a)==pytest.approx(1)
    y=torch.randn(20,10)
    assert 0<=dyn.linear_cka_gram(a,dyn.centered_gram(y))<=1
    assert dyn.linear_cka_gram(np.zeros((20,20)),a)==0
    with pytest.raises(ValueError): dyn.centered_gram(torch.zeros(2,4))


@pytest.fixture(scope='module')
def v04(tmp_path_factory):
    path=tmp_path_factory.mktemp('dynamics')
    m=run_research(name='tiny-zf',epochs=2,train_limit=128,val_limit=40,
                   interval=1,out=str(path))
    original=dyn.BASE_DIR
    dyn.BASE_DIR=path
    for f in [dyn.load_run,dyn.probe_data,dyn.model_at,dyn.activations_at,dyn.activation_gram]: f.cache_clear()
    yield m,TestClient(app),path
    dyn.BASE_DIR=original
    for f in [dyn.load_run,dyn.probe_data,dyn.model_at,dyn.activations_at,dyn.activation_gram]: f.cache_clear()


def test_v04_real_recorded_training(v04):
    m,_,path=v04
    assert m['provenance']['interpolated'] is True
    assert m['provenance']['native_resolution']==[8,8]
    assert m['provenance']['model_resolution']==[32,32]
    assert len(m['events'])==4
    assert len(m['snapshots'])==5
    assert m['snapshots'][0]['step']==0
    assert {r['step'] for r in m['snapshots']}=={0,1,2,3,4}
    assert all(e['gradient_global_preclip']>0 for e in m['events'])
    assert all(e['gradient_histogram'] is not None for e in m['events'])
    assert all(e['update_global_l2']>0 for e in m['events'])
    init=torch.load(path/'tiny-zf'/'step_000000.pt',weights_only=True)
    last=torch.load(path/'tiny-zf'/'step_000004.pt',weights_only=True)
    assert not torch.equal(init['conv1.weight'],last['conv1.weight'])


def test_v04_cka_symmetric_diag_same_probe(v04):
    meta,client,_=v04
    response=client.get('/api/dynamics/cka',params={'run_id':'tiny-zf','layer':4})
    assert response.status_code==200,response.text
    obj=response.json();matrix=np.array(obj['matrix'])
    assert matrix.shape==(5,5)
    assert np.allclose(matrix,matrix.T,atol=1e-6)
    assert np.allclose(np.diag(matrix),1,atol=1e-6)
    assert np.all((matrix>=0)&(matrix<=1))
    assert obj['probe_count']==40


def test_v04_snapshots_clusters_and_probe(v04):
    meta,client,_=v04
    listing=client.get('/api/dynamics/runs').json()
    assert listing['runs'][0]['run_id']=='tiny-zf'
    snap=client.get('/api/dynamics/snapshot',params={'run_id':'tiny-zf','step':4,'layer':3})
    assert snap.status_code==200
    assert sum(snap.json()['weight_histogram']['counts'])>0
    assert sum(snap.json()['gradient_histogram']['counts'])>0
    clusters=client.get('/api/dynamics/clusters',params={'run_id':'tiny-zf','step':4,'layer':3,'groups':4})
    assert clusters.status_code==200,clusters.text
    obj=clusters.json()
    assert sum(obj['counts'])==48
    assert len(obj['channels'])==48
    pair=client.get('/api/dynamics/probe',params={'run_id':'tiny-zf','step_a':0,'step_b':4,
                               'layer':3,'channel':0,'sample_index':0})
    assert pair.status_code==200,pair.text
    data=pair.json()
    assert data['activation_a'].startswith('data:image/png;base64,')
    assert data['difference'].startswith('data:image/png;base64,')
    assert data['rms_difference']>0
    assert data['scale']['scope']=='probe_layer_all_steps_all_channels'
    assert data['cka']==pytest.approx(client.get('/api/dynamics/cka',params={'run_id':'tiny-zf','layer':3}).json()['matrix'][0][-1],abs=1e-6)


def test_v04_invalid_requests_and_files(v04):
    _,client,_=v04
    assert client.get('/api/dynamics/manifest',params={'run_id':'../secret'}).status_code==404
    assert client.get('/api/dynamics/cka',params={'run_id':'tiny-zf','layer':6}).status_code==422
    assert client.get('/api/dynamics/snapshot',params={'run_id':'tiny-zf','step':22,'layer':1}).status_code==400
    assert client.get('/api/dynamics/probe',params={'run_id':'tiny-zf','step_a':0,'step_b':4,'layer':1,'channel':900,'sample_index':0}).status_code==400
    assert client.get('/dynamics.css').status_code==200
    assert client.get('/dynamics.js').status_code==200
    assert b'id="dynamics"' in client.get('/').content