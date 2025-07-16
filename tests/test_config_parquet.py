import pytest
from chatwhatcartobuy.config.parquet_config import _load_schema, get_submission_schema, \
    get_comment_schema, get_parquet_configs
from unittest.mock import mock_open, patch
from pathlib import Path
from pyarrow import string as pa_string, field as pa_field
import json

mock_schema = {
  "submission": {
    "submission_id": "pa.string()",
    "title": "pa.string()",
    "selftext": "pa.string()",
    "score": "pa.int64()",
    "upvote_ratio": "pa.float64()",
    "timestamp": "pa.timestamp('s')",
    "subreddit": "pa.string()",
    "num_comments": "pa.int64()"
  },
  "comment": {
    "comment_id": "pa.string()",
    "body": "pa.string()",
    "score": "pa.int64()",
    "timestamp": "pa.timestamp('s')",
    "subreddit": "pa.string()",
    "parent_submission_id": "pa.string()"
  }
}

@pytest.fixture
def mock_valid_path(monkeypatch):
    """Avoids Exception raises from load_schema"""
    monkeypatch.setattr(Path, 'exists', lambda self: True)
    return Path()

def test_load_submission_schema(mock_valid_path, monkeypatch):
    """Test if submission schema returned with valid file argument."""
    mock_path = mock_valid_path
    # Patch builtin open function to return mocked schema as json file
    # Read submission schema
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        schema = _load_schema(mock_path, 'submission')
        # Check that all column names were loaded as pa.Schema
        assert set(schema.names) == set(mock_schema['submission'])
        # Check that metadata was inserted
        assert schema.metadata == {b'record_type':b'submission'}
        # Check if first column corresponds to a pyarrow field expression for dtype
        assert schema[0] == pa_field('submission_id', pa_string())
    
def test_load_comment_schema(mock_valid_path, monkeypatch):
    """Test if comment schema returned with valid file argument."""
    mock_path = mock_valid_path
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        schema = _load_schema(mock_path, 'comment')
        assert set(schema.names) == set(mock_schema['comment'])
        assert schema.metadata == {b'record_type':b'comment'}
        assert schema[0] == pa_field('comment_id', pa_string())
    
def test_case_insensitive_load(mock_valid_path):
    """Test if main object and child object are converted to lowercase when loaded."""
    mock_schema = {'SUBMISSION': {'SUBMISSION_ID':'pa.string()'}}
    mock_path = mock_valid_path
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        schema = _load_schema(mock_path, 'submission')
        assert schema.names[0] == 'submission_id'
        assert schema.metadata[b'record_type'] == b'submission'

def test_load_schema_invalid_path():
    """Test if ValueError raised when file doesn't exist."""
    with pytest.raises(ValueError):
        _load_schema('invalid_fp.json', 'submission')
        
def test_default_submission_schema():
    """Test default values from get_submission_schema."""
    schema = get_submission_schema()
    assert set(schema.names) == set(mock_schema['submission'])
    assert schema.metadata == {b'record_type':b'submission'}
    assert schema[0] == pa_field('submission_id', pa_string())
    
def test_json_submission_schema(mock_valid_path):
    """Test when schemas.json file passed as kwarg to get_submission_schema func."""
    mock_path = mock_valid_path
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        schema = get_submission_schema(mock_path)
        assert schema.metadata == {b'record_type':b'submission'}
        assert schema[0] == pa_field('submission_id', pa_string())

def test_default_comment_schema():
    """Test default values from get_comment_schema."""
    schema = get_comment_schema()
    assert set(schema.names) == set(mock_schema['comment'])
    assert schema.metadata == {b'record_type':b'comment'}
    assert schema[0] == pa_field('comment_id', pa_string())

def test_json_comment_schema(mock_valid_path):
    """Test when schemas.json file passed as kwarg to get_comment_schema func."""
    mock_path = mock_valid_path
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        schema = get_comment_schema(mock_path)
        assert schema.metadata == {b'record_type':b'comment'}
        assert schema[0] == pa_field('comment_id', pa_string())
        
def test_invalid_dtype(mock_valid_path):
    """Test when schemas.json file is valid but contains invalid dtypes for pa.Schema."""
    mock_path = mock_valid_path
    mock_schema = {'submission': {'submission_id': 'string'}}
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
        # eval() should raise a value error since 'string' was passed
        with pytest.raises(NameError):
            get_submission_schema(mock_path)
            
def test_get_default_configs():
    """Tests if default config is returned."""
    root = Path('UsedCarAnalytics')
    submission_config, comment_config = get_parquet_configs(root)
    # Check if record types are correct
    assert submission_config.record_type == 'submission'
    assert comment_config.record_type == 'comment'
    # Check if dataset paths are correct
    assert submission_config.dataset_path == root / 'data/raw' / 'submission-dataset'
    assert comment_config.dataset_path == root / 'data/raw' / 'comment-dataset'
    # Check if schemas are correct
    assert submission_config.schema == get_submission_schema()
    assert comment_config.schema == get_comment_schema()

def test_get_custom_configs():
    """Tests when custom arguments are passed to function."""
    root = Path('UCA')
    # Construct with string
    submission_config, comment_config = get_parquet_configs(
        root, subdir='datasets', dataset_dirs=('sub','com')
        )
    # Check if dataset paths have been customized
    assert submission_config.dataset_path == root / 'datasets' / 'sub'
    assert comment_config.dataset_path == root / 'datasets' / 'com'
    
def test_invalid_dataset_dirs():
    """Tests if error raised when length of dataset dirs is != 2."""
    with pytest.raises(ValueError):
        get_parquet_configs(dataset_dirs=(1,2,3,4))
    with pytest.raises(ValueError):
        get_parquet_configs(dataset_dirs=('a',))
    
    