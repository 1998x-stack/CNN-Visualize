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
    server.fixed_activation_scale.cache_clear()
    server.step_timeline.cache_clear()
    server.step_checkpoint.cache_clear()
    server.step_activation_scale.cache_clear()
    yield manifest, TestClient(server.app)
    server.context.cache_clear()
    server.checkpoint.cache_clear()
    server.fixed_activation_scale.cache_clear()
    server.step_timeline.cache_clear()
    server.step_checkpoint.cache_clear()
    server.step_activation_scale.cache_clear()


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
    channel_track = client.get('/api/channel-trajectory',params={'sample_id':sample,'layer':2,'channel':3})
    assert channel_track.status_code == 200
    assert len(channel_track.json()['history']) == 3
    assert channel_track.json()['history'][0]['activation_rms_change'] == 0.0
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


def test_training_gradient_statistics(experiment):
    manifest, _ = experiment
    assert manifest['gradient_norm_definition'].startswith('Global L2 norm before clipping')
    for record in manifest['metrics'][1:]:
        assert record['grad_norm_batches'] == 2  # 180 samples, minibatch size 128
        assert record['grad_norm_max'] >= record['grad_norm_mean'] > 0
        assert record['grad_norm_max'] >= record['grad_norm_last'] > 0
        # Retain the pre-v0.2.1 field for existing consumers; it is LAST batch, not mean.
        assert record['grad_norm'] == record['grad_norm_last']
    assert manifest['metrics'][0]['grad_norm_mean'] is None


def test_fixed_target_class_and_same_prediction(experiment):
    manifest, client = experiment
    sample = manifest['samples'][0]['id']
    common = {'sample_id': sample, 'epoch': 2, 'layer': 3, 'channel': 0}
    zero = client.get('/api/inspect', params={**common, 'target_class': 0}).json()
    one = client.get('/api/inspect', params={**common, 'target_class': 1}).json()
    assert zero['predicted_class'] == one['predicted_class']
    assert zero['attribution_target_class'] == 0
    assert one['attribution_target_class'] == 1
    assert zero['attribution_mode'] == one['attribution_mode'] == 'fixed'
    assert zero['attribution_target_probability'] != one['attribution_target_probability']
    assert zero['saliency'] != one['saliency']  # changing target must alter attribution
    assert zero['feature_maps'] == one['feature_maps']  # changing target cannot alter activations
    legacy = client.get('/api/inspect', params=common).json()
    assert legacy['attribution_mode'] == 'predicted'
    assert legacy['attribution_target_class'] == legacy['predicted_class']


def test_cross_epoch_fixed_color_scales_and_raw_difference(experiment):
    manifest, client = experiment
    sample = manifest['samples'][0]['id']
    base = {'sample_id': sample, 'layer': 2, 'channel': 3}
    first = client.get('/api/inspect', params={**base, 'epoch': 0}).json()
    last = client.get('/api/inspect', params={**base, 'epoch': 2}).json()
    other_channel = client.get('/api/inspect', params={**base, 'epoch': 1, 'channel': 5}).json()
    assert first['feature_scale'] == last['feature_scale'] == other_channel['feature_scale']
    assert first['feature_scale']['scope'] == 'sample_layer_all_epochs_all_channels'
    assert first['feature_scale']['high'] > 0
    pair = client.get('/api/compare', params={**base, 'epoch_a': 0, 'epoch_b': 2})
    assert pair.status_code == 200
    pair = pair.json()
    assert pair['scale'] == first['feature_scale']
    assert pair['a'] == first['feature'] and pair['b'] == last['feature']
    assert pair['difference'].startswith('data:image/png;base64,')
    from cnnviz.server import checkpoint, sample_tensor
    x, _ = sample_tensor(sample)
    with torch.no_grad():
        a = checkpoint(0).forward_trace(x)[1]['features'][1][0, 3]
        b = checkpoint(2).forward_trace(x)[1]['features'][1][0, 3]
        expected = b - a
    assert pair['delta_mean'] == pytest.approx(float(expected.mean()), abs=1e-6)
    assert pair['rms_difference'] == pytest.approx(float(expected.square().mean().sqrt()), abs=1e-6)
    assert pair['l1_mean_difference'] == pytest.approx(float(expected.abs().mean()), abs=1e-6)
    assert pair['difference_scale']['high'] == pytest.approx(pair['scale']['high'])
    self_pair = client.get('/api/compare', params={**base, 'epoch_a': 1, 'epoch_b': 1}).json()
    assert self_pair['rms_difference'] == 0
    assert self_pair['delta_mean'] == 0
    assert self_pair['max_absolute_difference'] == 0
    assert self_pair['cosine_similarity'] is None or self_pair['cosine_similarity'] == pytest.approx(1.0, abs=1e-6)


def test_local_mode_and_invalid_comparisons(experiment):
    manifest, client = experiment
    sample = manifest['samples'][0]['id']
    local = client.get('/api/inspect', params={'sample_id': sample, 'epoch': 1, 'scale_mode': 'local'}).json()
    assert local['feature_scale']['scope'] == 'per_tile_dynamic'
    pair = client.get('/api/compare', params={'sample_id': sample, 'epoch_a': 1, 'epoch_b': 2, 'scale_mode': 'local'}).json()
    assert pair['scale_mode'] == 'local'
    assert pair['difference_scale']['scope'] == 'fixed_symmetric_layer_scale'
    for path, params, expected_code in [
        ('/api/inspect', {'sample_id':sample, 'target_class':999},400),
        ('/api/inspect', {'sample_id':sample, 'scale_mode':'false'},422),
        ('/api/compare', {'sample_id':sample, 'epoch_b':999},400),
        ('/api/compare', {'sample_id':sample, 'layer':3, 'channel':200},400),
        ('/api/compare', {'sample_id':sample, 'scale_mode':'invalid'},422),
    ]:
        assert client.get(path, params=params).status_code == expected_code


def test_absolute_heatmap_scale_remains_fixed():
    from cnnviz.imaging import heatmap
    fixed1 = heatmap(np.array([[0., 1.]], dtype='float32'), low=0, high=10)
    fixed2 = heatmap(np.array([[0., 2.]], dtype='float32'), low=0, high=10)
    assert fixed1 != fixed2
    assert heatmap(np.array([[0., 1.]])) == heatmap(np.array([[0., 2.]]))  # local stretches
    assert heatmap(np.zeros((2,2)), signed=True, signed_limit=3).startswith('data:image/png;base64,')


def test_compare_before_inspect_does_not_poison_cached_model(experiment):
    # A checkpoint first loaded inside inference_mode() becomes an inference Tensor
    # and cannot later participate in autograd when computing Saliency/Grad-CAM.
    manifest, client = experiment
    import cnnviz.server as server
    sample = manifest['samples'][0]['id']
    server.checkpoint.cache_clear()
    compare = client.get('/api/compare', params={'sample_id': sample, 'epoch_a': 0,
                                                  'epoch_b': 2, 'layer': 3, 'channel': 1})
    assert compare.status_code == 200, compare.text
    info = client.get('/api/inspect', params={'sample_id': sample, 'epoch': 2,
                                               'layer': 3, 'channel': 1, 'target_class': 6})
    assert info.status_code == 200, info.text
    assert info.json()['attribution_target_class'] == 6


def test_step_events_are_real_postoptimizer_updates(experiment):
    manifest, client = experiment
    response = client.get('/api/steps')
    assert response.status_code == 200
    record = response.json()
    assert record['available'] and record['schema'] == 1
    assert [x['step'] for x in record['events']] == [1, 2, 3, 4]
    assert record['snapshots'][0]['step'] == 0
    assert [x['step'] for x in record['snapshots']] == [0, 2, 4]
    assert record['recording']['global_steps'] == 4
    import cnnviz.server as server
    for epoch, global_step in [(1, 2), (2, 4)]:
        recorded = server.step_checkpoint(global_step).state_dict()
        epoch_weights = server.checkpoint(epoch).state_dict()
        assert all(torch.equal(recorded[k], epoch_weights[k]) for k in recorded)
    for row in record['events']:
        assert row['gradient_norm_preclip'] > 0
        assert row['weight_delta_post_step'] > 0
        norm_by_layer = sum(float(v)**2 for v in row['gradient_norm_layers_preclip'].values()) ** 0.5
        assert norm_by_layer == pytest.approx(row['gradient_norm_preclip'], rel=0.0002)
        delta_by_layer = sum(float(v)**2 for v in row['weight_delta_layers'].values()) ** 0.5
        assert delta_by_layer == pytest.approx(row['weight_delta_post_step'], abs=1e-7)
        assert row['has_checkpoint'] == (row['step'] in (2,4))
    assert manifest['step_observatory']['saved_snapshots'] == 3


def test_step_probe_fixed_sample_class_and_signed_difference(experiment):
    manifest, client = experiment
    sample = manifest['samples'][0]['id']
    params = {'sample_id': sample, 'step_a': 0, 'step_b': 4,
              'layer': 2, 'channel': 3, 'target_class': 4}
    response = client.get('/api/step-probe', params=params)
    assert response.status_code == 200, response.text
    obj = response.json()
    assert obj['step_a'] == 0 and obj['step_b'] == 4
    assert obj['scale']['scope'] == 'sample_layer_all_saved_steps_all_channels'
    assert obj['a'].startswith('data:image/png;base64,')
    assert obj['b'].startswith('data:image/png;base64,')
    assert obj['difference'].startswith('data:image/png;base64,')
    assert obj['filter_delta'].startswith('data:image/png;base64,')
    assert obj['rms_difference'] > 0 and obj['kernel_delta_l2'] > 0
    assert obj['probability_delta'] == pytest.approx(obj['probability_b'] - obj['probability_a'], abs=1e-6)
    assert obj['logit_delta'] == pytest.approx(obj['logit_b'] - obj['logit_a'], abs=1e-6)
    import cnnviz.server as server
    x, _ = server.sample_tensor(sample)
    with torch.no_grad():
        ma = server.step_checkpoint(0)
        mb = server.step_checkpoint(4)
        la, ta = ma.forward_trace(x)
        lb, tb = mb.forward_trace(x)
        delta = tb['features'][1][0,3]-ta['features'][1][0,3]
    assert obj['delta_mean'] == pytest.approx(float(delta.mean()), abs=1e-7)
    assert obj['rms_difference'] == pytest.approx(float(delta.square().mean().sqrt()), abs=1e-7)
    assert obj['logit_b'] == pytest.approx(float(lb[0,4]), abs=1e-7)
    same = client.get('/api/step-probe',params={**params,'step_b':0}).json()
    assert same['rms_difference'] == 0
    assert same['kernel_delta_l2'] == 0
    assert same['probability_delta'] == 0
    assert len(obj['class_contributions']) == 8
    assert len(obj['top_changing_channels']) == 5
    assert all(x['activation_rms_difference'] >= y['activation_rms_difference']
               for x,y in zip(obj['top_changing_channels'],obj['top_changing_channels'][1:]))


def test_step_probes_reject_unrecorded_steps_and_invalid_inputs(experiment):
    manifest, client = experiment
    sample = manifest['samples'][0]['id']
    for params, code in [
        ({'sample_id':sample,'step_b':1},400),
        ({'sample_id':sample,'step_b':999},400),
        ({'sample_id':sample,'target_class':999},400),
        ({'sample_id':sample,'layer':3,'channel':999},400),
        ({'sample_id':'../../hello','step_b':2},400),
        ({'sample_id':sample,'step_b':-1},422),
    ]:
        assert client.get('/api/step-probe',params=params).status_code == code
    # Checkpoints first loaded under no_grad are compatible with later grad attribution.
    import cnnviz.server as server
    server.checkpoint.cache_clear()
    assert client.get('/api/step-probe',params={'sample_id':sample,'step_b':4}).status_code == 200
    assert client.get('/api/inspect',params={'sample_id':sample,'epoch':2,'target_class':2}).status_code == 200


def test_step_recording_can_be_disabled(tmp_path,monkeypatch):
    manifest = run_training(epochs=1,out=str(tmp_path),train_limit=128,record_steps=False)
    assert manifest['step_observatory']['available'] is False
    assert not (tmp_path / 'steps.json').exists()
    import cnnviz.server as server
    monkeypatch.setattr(server,'ARTIFACT_DIR',tmp_path)
    for cache in [server.context,server.checkpoint,server.fixed_activation_scale,server.step_timeline,
                  server.step_checkpoint,server.step_activation_scale]:
        cache.cache_clear()
    client=TestClient(server.app)
    response=client.get('/api/steps')
    assert response.status_code == 200 and response.json()['available'] is False
    assert client.get('/api/step-probe',params={'sample_id':manifest['samples'][0]['id']}).status_code==400
    for cache in [server.context,server.checkpoint,server.fixed_activation_scale,server.step_timeline,
                  server.step_checkpoint,server.step_activation_scale]:
        cache.cache_clear()