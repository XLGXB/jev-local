import copy
import sys
import types

import pytest

from jevlocal.integration.native_repository_navigation import (
    NativeRepositoryNavigationJudge,
    _validate_prepared,
)
import jevlocal.integration.native_repository_navigation as adapter


def prepared():
    return {'request_id': 'r1', 'hop_index': 1, 'query': 'find entry',
            'current_directory': '.', 'snapshot_sha256': 'a' * 64,
            'candidates': [{'id': 'f1', 'name': 'readme.md',
                            'relative_path': 'readme.md', 'kind': 'file'}]}


class Runtime:
    mode = 'native'
    backend = 'laya'
    fingerprint = 'b' * 64


class Predictor:
    runtime = Runtime()
    calibration = None
    metadata = {'representation_id': 'fixture'}

    def __init__(self, key='f1', fingerprint=None, calibrated=False):
        self.key, self.fp, self.calibrated = key, fingerprint or Runtime.fingerprint, calibrated
        self.seen = None

    def predict(self, request):
        self.seen = copy.deepcopy(request)
        keys = list(request['questions']['candidate']['criteria'])
        return {'runtime_fingerprint': self.fp,
                'answers': {'candidate': {'selected_key': self.key, 'calibrated': self.calibrated,
                                          'temperature': 1.0,
                                          'probabilities': {key: (1.0 if key == self.key else 0.0)
                                                            for key in keys},
                                          'p_selected': 1.0}},
                'inference_total_ms': 1}


def test_closed_choice_abstain_and_owned_binding():
    p = Predictor('__abstain__')
    judge = NativeRepositoryNavigationJudge(p)
    item = prepared()
    result = judge(item)
    assert result['selected_candidate_id'] is None
    assert set(p.seen['questions']['candidate']['criteria']) == {'f1', '__abstain__'}
    assert result['route']['rubric'] == 'repository-navigation-choice-v1'
    assert result['route']['semantic_quality'] == 'unqualified'
    assert result['route']['input_bindings'][0]['snapshot_sha256'] == item['snapshot_sha256']
    item['candidates'][0]['name'] = 'mutated'
    assert p.seen['state']['candidates'][0]['name'] == 'readme.md'


@pytest.mark.parametrize('change', [
    lambda x: x.update(hop_index=True),
    lambda x: x.update(snapshot_sha256='bad'),
    lambda x: x['candidates'][0].update(relative_path='../outside'),
    lambda x: x['candidates'][0].update(id='__abstain__'),
    lambda x: x.update(query='   '),
    lambda x: (x['candidates'][0].update(name='.'), x['candidates'][0].update(relative_path='.')),
    lambda x: (x['candidates'][0].update(name='..'), x['candidates'][0].update(relative_path='..')),
    lambda x: (x['candidates'][0].update(name='file:ads'), x['candidates'][0].update(relative_path='file:ads')),
    lambda x: (x['candidates'][0].update(name='CON.txt'), x['candidates'][0].update(relative_path='CON.txt')),
    lambda x: (x['candidates'][0].update(name='trail.'), x['candidates'][0].update(relative_path='trail.')),
    lambda x: x.update(current_directory='folder/../child'),
    lambda x: x.update(current_directory='folder:ads'),
])
def test_rejects_invalid_prepared_before_predict(change):
    item = prepared()
    change(item)
    p = Predictor()
    with pytest.raises(ValueError):
        NativeRepositoryNavigationJudge(p)(item)
    assert p.seen is None


def test_rejects_identity_and_calibration_mismatch():
    with pytest.raises(ValueError, match='identity changed'):
        NativeRepositoryNavigationJudge(Predictor(fingerprint='c' * 64))(prepared())
    with pytest.raises(ValueError, match='calibration'):
        NativeRepositoryNavigationJudge(Predictor(calibrated=True))(prepared())


def test_validator_enforces_prepared_byte_cap():
    item = prepared()
    item['query'] = 'x' * 8192
    _validate_prepared(item)
    item['candidates'] *= 15
    with pytest.raises(ValueError):
        _validate_prepared(item)


def test_generic_and_state_json_loaders_are_explicit_and_mutually_exclusive(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'state_representation.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(adapter, 'validate_native_spec', lambda spec: {'checkpoint': str(checkpoint)})
    called = []
    monkeypatch.setattr(adapter, 'build_native_predictor', lambda spec: called.append('generic') or Predictor())
    with pytest.raises(ValueError, match='explicit state-json route'):
        adapter.build_native_repository_navigation_judge({})
    assert called == []
    fake_module = types.ModuleType('jevlocal.integration.represented_native')
    fake_module.build_predictor = lambda config: called.append('state-json') or Predictor()
    monkeypatch.setitem(sys.modules, 'jevlocal.integration.represented_native', fake_module)
    judge = adapter.build_state_json_repository_navigation_judge({})
    assert isinstance(judge, NativeRepositoryNavigationJudge)
    assert called == ['state-json']
