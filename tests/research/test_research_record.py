import json

import pandas as pd

from asharerank import research_record


def test_legacy_provenance_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(research_record, '_inputs', {})
    path = tmp_path / 'cache.parquet'
    frame = pd.DataFrame({'date': ['2020-01-01'], 'close': [10.]})
    frame.to_parquet(path)
    research_record.register_input(path, frame)
    manifest = research_record.input_manifest()
    assert not manifest['provenance_complete_for_registered_files']
    assert manifest['files'][0]['unknown_source_rows'] == 1
    assert manifest['files'][0]['data_kind'] == 'unknown'


def test_cli_paths_are_json_serializable(tmp_path, monkeypatch):
    monkeypatch.setattr(research_record, '_arguments', {})
    research_record.record_arguments({'output': tmp_path, 'seed': 17})
    assert json.loads(json.dumps(research_record._arguments))['seed'] == 17
