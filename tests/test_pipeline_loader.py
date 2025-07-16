import pytest
import pyarrow as pa
from usedcaranalytics.pipeline.loader import ParquetDataLoader
from usedcaranalytics.config.parquet_config import ParquetConfig
from unittest.mock import patch

@pytest.fixture
def mock_get_parquet_configs(tmp_path_factory):
    submission_schema = pa.schema([
        ("submission_id", pa.string()),
        ("title", pa.string())
        ])
    comment_schema = pa.schema([
        ("comment_id", pa.string()),
        ("body", pa.string())
        ])
    temp_dir = tmp_path_factory.mktemp('test_loader') / 'data/raw'
    sub_cfg = ParquetConfig('submission', temp_dir / 'submission-dataset', submission_schema)
    com_cfg = ParquetConfig('comment', temp_dir / 'comment-dataset', comment_schema)
    return (sub_cfg, com_cfg)

def test_initialization(mock_get_parquet_configs):
    """Test ParquetDataLoader constructor."""
    def simple_transform(table): return table
    fake_loader = ParquetDataLoader(mock_get_parquet_configs, transformer=simple_transform)
    assert fake_loader.config == mock_get_parquet_configs
    assert callable(fake_loader._transformer)

def test_init_invalid_config(mock_get_parquet_configs):
    """Test exceptions raised for invalid config"""
    sub_cfg, _ = mock_get_parquet_configs
    # Single namedtuple instead of two
    with pytest.raises(ValueError):
        ParquetDataLoader((sub_cfg,), target_MB=16)
    # Incorrect type (non-namedtuple)
    with pytest.raises(AttributeError):
        ParquetDataLoader(('invalid', 'config'), target_MB=16)
    # Negative MB value
    with pytest.raises(ValueError):
        ParquetDataLoader(mock_get_parquet_configs, target_MB=-5)
    # Zero MB value    
    with pytest.raises(ValueError):
        ParquetDataLoader(mock_get_parquet_configs, target_MB=-5)
    # String MB value
    with pytest.raises(TypeError):
        ParquetDataLoader(mock_get_parquet_configs, target_MB='invalid') 

def test_configure_loader(mock_get_parquet_configs):
    """Test if named values from config tuple are loaded as instance vars."""
    loader = ParquetDataLoader(mock_get_parquet_configs)
    assert loader._schemas == {
        'submission': mock_get_parquet_configs[0].schema,
        'comment': mock_get_parquet_configs[1].schema
        }
    assert loader._dataset_paths == {
        'submission': mock_get_parquet_configs[0].dataset_path,
        'comment': mock_get_parquet_configs[1].dataset_path
        }
    assert loader._buffers == {
        'submission': {'submission_id': [], 'title': []},
        'comment': {'comment_id': [], 'body': []}
        }
    assert loader._buffer_sizes == {'submission': 0, 'comment': 0}
    assert loader._batch_counters == {'submission': 0, 'comment': 0}

def test_set_target_mb(mock_get_parquet_configs):
    """Test if target_MB setter works and if the _target_bytes is automatically set."""
    # Check if it works on initialization
    fake_loader = ParquetDataLoader(mock_get_parquet_configs, target_MB=16)
    assert fake_loader.target_MB == 16
    # Check if it works post initialization
    fake_loader.set_target_mb(32.5)
    assert fake_loader.target_MB == 32.5
    assert fake_loader._target_bytes == int(32.5 * 2**20)

@patch('usedcaranalytics.pipeline.loader.ParquetDataLoader._flush')
def test_load(stubbed_flush, mock_get_parquet_configs):
    """Test loading logic with mocked data_stream."""
    # Get temporary dataset paths
    sub_cfg, com_cfg = mock_get_parquet_configs
    sub_path, com_path = sub_cfg.dataset_path, com_cfg.dataset_path
    
    # When called, return a flushed buffer and byte count
    stubbed_flush.side_effect = lambda buffer, *args, **kwargs: ({k: [] for k in buffer}, 0)
    
    # target_MB==1 such that it will call _write for every record
    loader = ParquetDataLoader(mock_get_parquet_configs, target_MB=1 * 2**-20)
    fake_stream = iter([
        ('submission', {'submission_id': 'sid1', 'title': 't1'}),
        ('comment', {'comment_id': 'cid1', 'body': 'b1'}),
        ('comment', {'comment_id': 'cid10', 'body': 'b10'}),
        ('comment', {'comment_id': 'cid15', 'body': 'b15'}),
        ('submission', {'submission_id': 'sid2', 'title': 't2'})
        ])
    
    loader.load(fake_stream)
    
    # Should call _write for every record while streaming, then 1 call per 
    # record type at the end of load() -> 7 calls
    assert loader._flush.call_count == 7
    # Files must at least be the number of respective records for sub & com data
    assert len(list(sub_path.glob('*.parquet'))) >= 2
    assert len(list(com_path.glob('*.parquet'))) >= 3

# Pytest tmp_path fixture creates a temp directory to simulate disk write
def test_write(tmp_path, mock_get_parquet_configs):
    """Test _write method with temporary path."""
    loader = ParquetDataLoader(mock_get_parquet_configs)
    loader._dataset_paths['submission'] = tmp_path
    buffer = {'submission_id': ['sid1'], 'title': ['t1']}
    loader._write('submission', buffer)
    # Should write exactly one file to temp directory
    exported_files = list(tmp_path.glob('submission-*.parquet'))
    assert len(exported_files) == 1
    # Read parquet table and ensure written data match buffer
    read_table = pa.parquet.read_table(exported_files[0])
    assert read_table.column('submission_id').to_pylist() == ['sid1']
    assert read_table.column('title').to_pylist() == ['t1']

def test_flush():
    """Test buffer flushing method."""
    # Create a buffer and ensure it returns a tuple (dict of empty lists, 0)
    buffer = {'col1': [1, 2], 'col2': ['a', 'b']}
    flushed_buffer, flushed_byte_count = ParquetDataLoader._flush(buffer)
    assert flushed_buffer == {'col1': [], 'col2': []}
    assert flushed_byte_count == 0

