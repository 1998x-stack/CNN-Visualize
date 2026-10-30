import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from cnnviz.model import InspectableCNN, deconv_projection
from cnnviz.data import get_dataset
from cnnviz.imaging import heatmap, input_image
from cnnviz.train import run_training


def test_model_trace_and_projection():
    model = InspectableCNN()
    x = torch.rand(1, 1, 16, 16)
    logits, trace = model.forward_trace(x)
    assert logits.shape == (1, 10)
    assert [list(t.shape) for t in trace['features']] == [[1, 12, 16, 16], [1, 20, 8, 8], [1, 28, 4, 4]]
    for layer in (1, 2, 3):
        result = deconv_projection(model, trace, layer, 0)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    with pytest.raises(ValueError):
        deconv_projection(model, trace, 4, 0)


def test_digits_data_and_images():
    x, y, train, val, classes, source = get_dataset('digits')
    assert x.shape == (1797, 1, 16, 16)
    assert len(train) + len(val) == 1797
    assert not set(train).intersection(val)
    assert classes == list('0123456789')
    assert input_image(x[0]).startswith('data:image/png;base64,')
    assert heatmap(x[0,0]).startswith('data:image/png;base64,')


@pytest.fixture(scope='function')
def experiment(tmp_path, monkeypatch):
    folder = tmp_path
    manifest = run_training(epochs=2, out=str(folder), train_limit=180)
    import cnnviz.server as server
    monkeypatch.setattr(server, 'ARTIFACT_DIR', folder)
    server.context.cache_clear()
    server.checkpoint.cache_clear()
    yield manifest, TestClient(server.app)
    server.context.cache_clear()
    server.checkpoint.cache_clear()


def test_training_checkpoints_and_api(experiment):
    manifest, client = experiment
    assert [m['epoch'] for m in manifest['metrics']] == [0,1,2]
    assert manifest['samples']
    assert client.get('/api/health').json()['has_artifacts']
    data = client.get('/api/manifest').json()
    assert len(data['samples']) == 40
    sample = data['samples'][0]['id']
    trajectory = client.get('/api/trajectory',params={'sample_id':sample}).json()
    assert len(trajectory['evolution']) == 3
    inspected = client.get('/api/inspect',params={'sample_id':sample,'epoch':2,'layer':2,'channel':3})
    assert inspected.status_code==200, inspected.text
    info = inspected.json()
    assert len(info['feature_maps'])==20
    assert len(info['probabilities'])==10
    assert abs(sum(info['probabilities'])-1)<1e-4
    assert info['deconv'].startswith('data:image/png;base64,')
    assert client.get('/api/inspect',params={'sample_id':sample,'epoch':90}).status_code==400
    assert client.get('/api/inspect',params={'sample_id':'999999'}).status_code==400
    assert client.get('/').status_code==200
    assert client.get('/app.css').status_code==200
    assert client.get('/app.js').status_code==200


def test_upload(experiment):
    _, client = experiment
    from PIL import Image
    import io
    buffer=io.BytesIO();Image.new('RGB',(128,128),'white').save(buffer,'PNG')
    upload = client.post('/api/upload',files={'file':('sample.png',buffer.getvalue(),'image/png')})
    assert upload.status_code==200, upload.text
    info = upload.json()
    prediction = client.get('/api/inspect',params={'sample_id':info['id'],'epoch':0})
    assert prediction.status_code==200
    assert prediction.json()['correct'] is None
    assert client.post('/api/upload',files={'file':('bad.png',b'noimage','image/png')}).status_code==415