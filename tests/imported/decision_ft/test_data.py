import copy
import json
import pytest
from decision_ft.data import (atomize, target_vector, public_request, permutation,
    validate_record, partition_rows, write_jsonl, read_jsonl, _leakage_keys)
from decision_ft.make_demo import rows


def test_noul_order():
    keys, target, label = target_vector({'type': 'noul', 'label': True})
    assert keys == ['false', 'true'] and target == [0., 1.] and label == 1


@pytest.mark.parametrize('bad', ['false', 0, 1, None])
def test_noul_rejects_non_boolean(bad):
    with pytest.raises(ValueError):
        target_vector({'type': 'noul', 'label': bad})


@pytest.mark.parametrize('bad', [-1, 2, True, 0.5])
def test_score_rejects_invalid_level(bad):
    with pytest.raises(ValueError):
        target_vector({'type': 'score', 'criteria': ['low', 'high'], 'label': bad})


@pytest.mark.parametrize('target', [{'a': 0.8}, {'a': 0.5, 'b': 0.9}, {'a': float('nan'), 'b': 0.1}, {'a': -0.1, 'b': 1.1}])
def test_invalid_soft_targets(target):
    with pytest.raises(ValueError):
        target_vector({'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}, 'label': 'a', 'target': target})


def test_no_label_or_metadata_in_model_input():
    row = rows(1)[0]
    row['_meta']['secret_after_action'] = 'do-not-encode'
    row['questions']['next']['target'] = {'open_settings': 0.1, 'open_help': 0.8, 'needs_host': 0.1}
    raw = json.dumps(public_request(row))
    assert 'do-not-encode' not in raw and '"target"' not in raw and '"label"' not in raw
    for atom in atomize([row]):
        assert set(atom.question) <= {'type', 'instructions', 'criteria'}


def test_target_alignment_after_choice_permutation():
    row = rows(1)[0]
    row['questions']['next']['target'] = {'open_settings': 0.1, 'open_help': 0.8, 'needs_host': 0.1}
    original_score = row['questions']['progress']['criteria'][:]
    for seed in range(8):
        shuffled = permutation(row, seed)
        atom = atomize([shuffled])[0]
        assert dict(zip(atom.keys, atom.target)) == row['questions']['next']['target']
        assert atom.keys[atom.label] == 'open_help'
        assert shuffled['questions']['progress']['criteria'] == original_score


def test_authorization_required():
    row = rows(1)[0]
    row['_meta']['training_allowed'] = False
    with pytest.raises(ValueError):
        validate_record(row)


def test_grouped_split_deterministic_and_no_leakage():
    samples = rows(40)
    duplicate = copy.deepcopy(samples[0])
    duplicate['_meta']['id'] = 'sibling'
    duplicate['_meta']['group_id'] = 'another-group'
    samples.append(duplicate)
    splits = partition_rows(samples, 7)
    assert splits == partition_rows(samples, 7)
    owner = {}
    for name, part in splits.items():
        for row in part:
            for key in _leakage_keys(row):
                assert key not in owner or owner[key] == name
                owner[key] = name


def test_shared_templates_join_components():
    samples = rows(40)
    samples[0]['_meta']['template_id'] = samples[1]['_meta']['template_id'] = 'same-template'
    parts = partition_rows(samples)
    find = lambda uid: next(n for n, rs in parts.items() if any(r['_meta']['id'] == uid for r in rs))
    assert find('demo-0000') == find('demo-0001')


def test_json_roundtrip(tmp_path):
    path = tmp_path / 'rows.jsonl'
    write_jsonl(path, rows(4))
    assert read_jsonl(path) == rows(4)


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"state":1,"state":2}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        read_jsonl(path)
